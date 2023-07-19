import os
import numpy as np
import torch.optim as optim
from model.config import get_args
from model.DQNSearchAgent import Agent, ReplayBuffer,DQN
import warnings
from model.nwp import set_seed
from dateutil import tz
import torch
import math
import string
import logging
import random
from transformers import RobertaTokenizer, RobertaForMaskedLM

tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_reward(states, input_letter=None):
    rewards=[1/(1.5*len(state)-state.count(input_letter)) for state in states]
    rewards=np.asarray(rewards)
    reward=np.max(rewards)
    best_cand_state=states[np.argmax(rewards)]

    return reward, best_cand_state
    # return state.count('a')

# Function to edit a state with a given action and position, and an optional input letter
def edit(state, action,position,input_letter=None):
    if action == 0: # replace
        state = state[:position] + input_letter + state[position + 1:]
    elif action == 1: # insert
        state = state[:position] + input_letter + state[position:]
    elif action == 2: # delete
        state = state[:position] + state[position + 1:]
    return state

# Function to load data from a given input file and log file
def load_data(input_file, infer_file):
    with open(input_file, 'r', encoding='utf8') as f:
        data = f.readlines()
        test_data = [line.strip() for line in data]

    log_txt_path=os.path.join(infer_file.split('.txt')[0] + '.log')
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='',filename=log_txt_path,filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logging.info(args)

    return test_data

# Function to initialize the model, network, replay buffer, optimizer and epsilon
def initialize_model(args):
    rbt_model = RobertaForMaskedLM.from_pretrained('roberta-large', return_dict=True).to(device)
    rbt_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    agent = Agent(args, device, rbt_model, rbt_tokenizer).to(device)

    # Initialize the network
    local_net = DQN(agent.state_dim, args.num_actions).to(device)
    target_net = DQN(agent.state_dim, args.num_actions).to(device)
    replay_buffer = ReplayBuffer(args.buffer_size)

    # Initialize the optimizer
    optimizer = optim.Adam(local_net.parameters(), lr=args.lr)

    # epsilon-exploration for choosing an action
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 200

    epsilons = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    return agent, local_net, target_net, replay_buffer, optimizer, epsilons

# generate am action given a state
def perform_action(agent,state, local_net, epsilon):

    with torch.no_grad():
        actions = agent.act(state, epsilon, local_net)

        # if the action is delete, we need to make sure the state is not empty
        while len(state[0]) == 1 and actions.item() == 2:
            actions = agent.act(state, epsilon, local_net)

    return actions

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

    # ------------------- soft update target network ------------------- #
    global_step += 1
    if global_step % args.update_interval == 0:
        """  Soft update model parameters.
             θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_net.parameters(),
                                             local_net.parameters()):
            target_param.data.copy_(args.tau * local_param.data + (1 - args.tau) * target_param.data)
        torch.save(target_net.state_dict(),
                   os.path.join(output_dir, str(global_step) + '_target_net.pt'))

    return loss.item(), global_step

# Function to create reference news given a state and actions
def create_ref_news(state, actions, MAX_LEN):
    ref_news = []
    cur_state = state[0][:MAX_LEN]
    action = actions[0]

    if action != 2:
        for position in reversed(range(len(state))):
            for cand_letter in list(string.ascii_lowercase):
                edited_state = edit(cur_state, action, position, cand_letter)[:MAX_LEN]
                ref_news.append(edited_state)
    else:
        for position in reversed(range(len(state))):
            edited_state = edit(cur_state, action, position)[:MAX_LEN]
            ref_news.append(edited_state)

    return ref_news

def update_replay_buffer_and_state(replay_buffer, state, actions, max_episode_reward, reward, best_cand_state, done):
    accept=False
    if reward > max_episode_reward:
        max_episode_reward = reward
        replay_buffer.push(state[0], actions[0], max_episode_reward, best_cand_state,done)
        state = [best_cand_state]
        accept=True
    else:
        state=state
    return replay_buffer, state, max_episode_reward, accept


def train(args,output_dir):

    # Setting seed for reproducibility
    set_seed(args.seed)
    BSZ = args.bsz
    input_file = 'data/toy_train.txt'
    infer_file = output_dir + '_output.txt'
    MAX_LEN = 10

    # load data
    test_data=load_data(input_file, infer_file)

    # load model
    agent, local_net, target_net, replay_buffer, optimizer, epsilons = initialize_model(args)

    # start inference
    print("start inference...")

    with open(infer_file, 'w', encoding='utf8') as f:

        global_step=0
        torch.cuda.empty_cache()

        # batch inference
        for idx,batch_data in enumerate(test_data):

            # generate a random input letter
            input_letter = random.choice(string.ascii_lowercase)
            ref_olds=batch_data

            state=[ref_olds]

            max_episode_reward = float('-inf')
            losses = []

            for step in range(args.max_steps):


                # infer actions
                epsilon = epsilons(idx)
                actions = perform_action(agent, state, local_net, epsilon)

                ref_news=create_ref_news(state, actions, MAX_LEN)

                done = True if step == args.max_steps - 1 else False

                # get max reward
                cur_state=state[0]
                reward, best_cand_state=get_reward(ref_news, input_letter)

                replay_buffer, state, max_episode_reward, accept = update_replay_buffer_and_state(replay_buffer, state, actions,
                                                                                          max_episode_reward, reward,
                                                                                          best_cand_state,done)
                # ------------------- update Q network ------------------- #
                if len(replay_buffer) >= args.buffer_size:
                    loss, global_step=update_networks(agent, replay_buffer, local_net, target_net, optimizer, args, global_step,losses)

                    print("loss is {:.6f}\t reward is {:.6f}, old_state is {}\taction is {},\tletter I wanna have is '{}'\t"
                          "the new state is {}\t accept is {}"
                          .format(loss, reward, cur_state, str(actions[0].item()), input_letter, state[0], accept))
                    logging.info("loss is {:.6f}\t reward is {:.6f}, old_state is {}\taction is {},\tletter I wanna have is '{}'"
                                 "\tthe new state is {}\t accept is {}"
                          .format(loss, reward, cur_state, str(actions[0].item() ), input_letter, state[0], accept))



            #update output.txt
            for i in range(BSZ):
                # print("original input is {}\tletter I wanna have is '{}'\tthe state is {}".format(batch_data, input_letter, state[0]))
                # logging.info("original input is {}\tletter I wanna have is '{}'\tthe state is {}".format(batch_data, input_letter, state[0]))
                f.write(state[0]+'\n')
                f.flush()




if __name__ == '__main__':
    args=get_args()
    output_dir = 'results/'+args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train(args,output_dir)
