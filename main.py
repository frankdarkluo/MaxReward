import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import itertools
import math
import logging
from model.DQNSearchAgent import Agent, ReplayBuffer,DQN
from model.Scorer import Scorer
from model.editor import RobertaEditor
from model.config import get_args
import torch.multiprocessing as mp
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

def train(args,train_set):
    set_seed(args.seed)
    editor = RobertaEditor(args,device).to(device)
    scorer= Scorer(args, editor,device).to(device)
    agent=Agent(editor,args,device).to(device)

    # Initialize the network
    local_net=DQN(agent.state_dim, args.num_actions).to(device)
    target_net = DQN(agent.state_dim, args.num_actions).to(device)
    replay_buffer = ReplayBuffer(args.buffer_size)

    # Initialize the optimizer

    optimizer = optim.Adam(local_net.parameters(), lr=args.lr)

    BSZ = args.bsz

    log_txt_path=os.path.join(of_dir, train_file.split('.txt')[0] + '.log')
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='',filename=log_txt_path,filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logging.info(args)

    # epsilon-exploration for choosing an action
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilons = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    # Initialize the train data
    train_data = DataLoader(train_set,
                            batch_size=args.bsz,
                            num_workers=2,
                            pin_memory=True,
                            drop_last=True)
    global_step = 0

    # training
    for idx,batch_data in enumerate(train_data):
        batch_data=sorted(batch_data, key=lambda x: len(x.split()), reverse=True)
        ref_olds=batch_data
        batch_state_vec, _ = editor.state_vec(batch_data)

        # Initialize the environment
        all_rewards = []
        losses = []

        state=ref_olds

        # start_time = time.time()

        # ------------------- train Q network ------------------- #
        max_episode_reward = [0 for _ in range(len(ref_olds))]
        for step in range(args.max_steps):

            torch.cuda.empty_cache()

            epsilon = epsilons(idx)
            actions = agent.act(state, epsilon,local_net)

            # ref_news = [editor.edit(state, [action] * BSZ, [positions] * BSZ) for positions in range(max_seq_len)]
            # ref_news=[sum([editor.edit([state[idx]], [action], [positions])[0] for positions in range(len(state[idx].split()))],[]) for idx in range(BSZ)]

            ref_news = []
            for idx in range(BSZ):
                state_words = state[idx].split()
                action=actions[idx]
                intermediate_results = []
                for positions in range(len(state_words)):
                    edited_state = editor.edit([state[idx]], [action], [positions])
                    intermediate_results+=edited_state

                ref_news.append(intermediate_results)

            if step<args.max_steps-1:
                done=False
            else: done=True # meaning when step=4, done=True

            # get reward
            results = [scorer.scoring(ref_news[i], [ref_olds[i]], [batch_state_vec[i]])
                                                                 for i in range(len(ref_news))]

            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print('代码执行时间：{:.6f}秒'.format(elapsed_time))

            index, ref_new_score, new_style_labels = zip(*results)

            temp_next_state = [ref_news[i][index[i]] for i in range(len(ref_news))]
            reward=list(ref_new_score)

            # ------------------- update states ------------------- #
            # if ref_new_score>ref_old_score and reward> max_episode_reward:
            for i in range(len(ref_news)):
                if reward[i]> max_episode_reward[i]:
                    max_episode_reward[i] = reward[i]
                    state[i] = temp_next_state[i]

        # ------------------- update replay buffer ------------------- #
        for i in range(BSZ):
            replay_buffer.push(state[i], actions[i], max_episode_reward[i], state[i], done)

        if done:
            all_rewards.append(max_episode_reward)

        # ------------------- update Q network ------------------- #
        if len(replay_buffer) >= args.buffer_size:
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
                torch.save(target_net.state_dict(), os.path.join(of_dir, str(global_step) + '_target_net.pt'))

        if idx % 1 == 0:
            plot(idx, all_rewards, losses)

    # infer(args, editor, scorer, agent)


if __name__ == '__main__':
    # Initialize the model
    args = get_args()
    set_seed(args.seed)

    of_dir = 'results/' + args.output_dir
    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')
    dst = args.dst

    train_file = '{}_{}_seed={}_{}_{}_{}.txt'. \
        format(timestamp, dst, str(args.seed), args.style_mode, str(args.style_weight), args.direction)

    train_dataset = TSTDataset(args, 'process_train')
    train(args,train_dataset)
