import os
import torch
import torch.optim as optim
import math
import logging
from model.DQNSearchAgent import Agent, ReplayBuffer,DQN
from model.Scorer import Scorer
from model.editor import RobertaEditor
from model.config import get_args
import warnings
from model.nwp import set_seed
import datetime
from dateutil import tz
from torch.utils.data import DataLoader, DistributedSampler
from utils.helper import plot, sync_initial_weights, get_free_port
from utils.dataset import TSTDataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import RobertaTokenizer, RobertaForMaskedLM


# Function to initialize the model, network, replay buffer, optimizer and epsilon
def initialize_model(args):
    
    rbt_model = RobertaForMaskedLM.from_pretrained('roberta-large', return_dict=True).to(device)
    rbt_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    print("loading roberta ...")

    editor = RobertaEditor(args, device, rbt_model, rbt_tokenizer).to(device)
    
    scorer = Scorer(args, editor, device).to(device)
    agent = Agent(args, device, rbt_model, rbt_tokenizer).to(device)

    # Initialize the network
    local_net = DQN(agent.state_dim, args.num_actions).to(device)
    target_net = DQN(agent.state_dim, args.num_actions).to(device)
    replay_buffer = ReplayBuffer(args.buffer_size)

    # Initialize the optimizer
    optimizer = optim.Adam(local_net.parameters(), lr=args.lr)

    # epsilon-exploration for choosing an action
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 1e3

    epsilons = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    return agent, scorer, editor, local_net, target_net, replay_buffer, optimizer, epsilons

# Function to load data from a given input file and log file
def load_data(train_set):
    log_txt_path = os.path.join(output_dir, train_file.split('.txt')[0] + '.log')
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='', filename=log_txt_path, filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.info(args)

    # Initialize the train data
    train_data = DataLoader(train_set,
                            batch_size=args.bsz,
                            num_workers=2,
                            pin_memory=True,
                            drop_last=True)
    return train_data

# Function to update networks
def update_networks(agent, replay_buffer, local_net, target_net, optimizer, args, global_step,losses):

    states, actions, rewards, next_states, dones = replay_buffer.sample(args.buffer_size)

    actions = torch.asarray(actions).to(device)
    rewards = torch.asarray(rewards).float().to(device)
    dones = torch.asarray(dones).to(device) + 0  # From T/F to be 0/1

    local_net.train()
    target_net.eval()
    q_values = local_net(agent.text2emb(states.tolist()))
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(agent.text2emb(next_states.tolist()))
        next_q_value = next_q_values.max(1)[0]

    # Max DQN
    expected_q_value = torch.max(rewards, next_q_value * (1 - dones))
    loss = (q_value - expected_q_value.data).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss is {}".format(str(loss.item())))
    logging.info("loss is {}".format(str(loss.item())))
    losses.append(loss.item())

    # ------------------- soft update target network ------------------- #
    global_step += 1
    if global_step % args.update_interval == 0:

        """  Soft update model parameters.
             θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_net.parameters(),
                                             local_net.parameters()):
            target_param.data.copy_(args.tau * local_param.data + (1 - args.tau) * target_param.data)
        torch.save(target_net.state_dict(), os.path.join(output_dir, str(global_step) + '_target_net.pt'))

    return loss.item(), global_step


def create_input_news(BSZ, state, actions, editor):
    # input_news = [editor.edit(state, [action] * BSZ, [positions] * BSZ) for positions in range(max_seq_len)]
    # input_news=[sum([editor.edit([state[idx]], [action], [positions])[0] for positions in range(len(state[idx].split()))],[]) for idx in range(BSZ)]
    input_news = []
    for idx in range(BSZ):
        state_words = state[idx].split()
        action = actions[idx]
        intermediate_results = []
        for positions in range(len(state_words)):
            edited_state = editor.edit([state[idx]], [action], [positions])
            intermediate_results += edited_state

        input_news.append(intermediate_results)

    return input_news

# Functions to get reward
def get_reward(results, input_news):

    index, input_new_score, new_style_labels = zip(*results)

    best_cand_states = [input_news[i][index[i]] for i in range(len(input_news))]
    reward = list(input_new_score)
    
    return reward, best_cand_states
    
def train(args,train_set):
    
    # initialize the environment
    agent, scorer, editor, local_net, target_net, replay_buffer, optimizer, epsilons = initialize_model(args)
    
    # load data
    train_data=load_data(train_set)

    global_step = 0

    # training
    print("start training...")
    
    for idx,batch_data in enumerate(train_data):
        batch_data=sorted(batch_data, key=lambda x: len(x.split()), reverse=True)
        input_olds=batch_data
        batch_state_vec, _ = editor.state_vec(batch_data)

        # Initialize the environment
        all_rewards = []
        losses = []

        state=input_olds

        # start_time = time.time()

        # ------------------- train Q network ------------------- #
        max_episode_reward = [0 for _ in range(len(input_olds))]
        epsilon = epsilons(idx)

        for step in range(args.max_steps):

            torch.cuda.empty_cache()

            actions = agent.act(state, epsilon,local_net)

            input_news=create_input_news(args.bsz, state, actions, editor)

            done = True if step == args.max_steps - 1 else False


            # get editing results
            results = [scorer.scoring(input_news[i], [input_olds[i]], [batch_state_vec[i]])
                       for i in range(len(input_news))]

            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print('代码执行时间：{:.6f}秒'.format(elapsed_time))

            # get reward
            reward, best_cand_states = get_reward(results, input_news)

            # update states and max_episode_reward
            for i in range(len(input_news)):
                if reward[i]> max_episode_reward[i]:
                    max_episode_reward[i] = reward[i]

                    # --------------- update replay buffer ------------------- #
                    replay_buffer.push(state[i], actions[i], reward[i], best_cand_states[i], done)
                    # ----------- update the state for next step ------------- #
                    state[i] = best_cand_states[i]

        if done:
            all_rewards.append(max_episode_reward)

        # ------------------- update Q network ------------------- #
        if len(replay_buffer) >= args.buffer_size:
            loss, global_step = update_networks(agent, replay_buffer, local_net, target_net, optimizer, args,
                                                global_step, losses)

        if idx % 1 == 0:
            plot(idx, all_rewards, losses)

    # infer(args, editor, scorer, agent)

if __name__ == '__main__':
    # Initialize the model
    args = get_args()
    set_seed(args.seed)

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')
    dst = args.dst

    output_dir = 'results/'+args.direction+'_'+timestamp+'_' + args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_file = '{}_{}.txt'. \
        format(dst, args.direction)

    train_dataset = TSTDataset(args, 'process_train')
    train(args,train_dataset)
