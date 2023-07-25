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
    """
    1. You get a positive reward for each occurrence of the target letter in the state.
    The more of these letters, the higher your reward.
    2.However, for each non-target letter in the state, you get a slight penalty (in this case, -0.1).
    This discourages unnecessary lengthening of the string with non-target letters,
    while still allowing flexibility for the necessary actions.
    """
    rewards = [1.1*state.count(input_letter) - 0.1 * (len(state) - state.count(input_letter)) for state in states]
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

# Function to check if a string is all letters
def is_all_letter(input_string, letter):
    return all(char == letter for char in input_string)

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
    epsilon_final = 0.2
    epsilon_decay = 1000

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

# Function to create inputerence news given a state and actions
def create_input_news(state, action, MAX_LEN):
    input_news = []
    cur_state = state[0][:MAX_LEN]

    if action != 2:
        for position in reversed(range(len(state[0]))):
            for cand_letter in list(string.ascii_lowercase):
                edited_state = edit(cur_state, action, position, cand_letter)[:MAX_LEN]
                input_news.append(edited_state)
    else:
        for position in reversed(range(len(state[0]))):
            edited_state = edit(cur_state, action, position)[:MAX_LEN]
            input_news.append(edited_state)

    return input_news

def update_replay_buffer_and_state(replay_buffer, state, action, max_episode_reward, reward, best_cand_state, done):
    accept=False
    if reward > max_episode_reward:
        max_episode_reward = reward
        replay_buffer.push(state[0], action, max_episode_reward, best_cand_state,done)
        state = [best_cand_state]
        accept=True
    else:
        state=state
    return replay_buffer, state, max_episode_reward, accept


def train(args,output_dir):

    # Setting seed for reproducibility
    # set_seed(args.seed)
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
            input_olds=batch_data

            state=[input_olds]

            max_episode_reward, _ = get_reward(state, input_letter)
            losses = []

            for step in range(args.max_steps):

                # infer actions
                # epsilon = epsilons(idx)
                # actions = perform_action(agent, state, local_net, epsilon)

                # create input news
                for action in range(3):
                    input_news=create_input_news(state, action, MAX_LEN)

                done = True if step == args.max_steps - 1 else False

                # get max reward
                cur_state=state[0]
                reward, best_cand_state=get_reward(input_news, input_letter)

                replay_buffer, state, max_episode_reward, accept = update_replay_buffer_and_state(replay_buffer, state, action,
                                                                                          max_episode_reward, reward,
                                                                                          best_cand_state,done)
                # ------------------- update Q network ------------------- #
                if len(replay_buffer) >= args.buffer_size:
                    loss, global_step=update_networks(agent, replay_buffer, local_net, target_net, optimizer, args, global_step,losses)

                    print("loss is {:.6f}\t reward is {:.6f}, old_state is {}\taction is {},\tletter I wanna have is '{}'\t"
                          "the new state is {}\t accept is {}"
                          .format(loss, reward, cur_state, str(action.item()), input_letter, state[0], accept))
                    logging.info("loss is {:.6f}\t reward is {:.6f}, old_state is {}\taction is {},\tletter I wanna have is '{}'"
                                 "\tthe new state is {}\t accept is {}"
                          .format(loss, reward, cur_state, str(action.item() ), input_letter, state[0], accept))

                if is_all_letter(state[0], input_letter):
                    print("I got the letter I want, the state is full of {} now.".format(input_letter))
                    logging.info("I got the letter I want, the state is full of {} now.".format(input_letter))
                    break

            #update output.txt
            for i in range(BSZ):
                # print("original input is {}\tletter I wanna have is '{}'\tthe state is {}".format(batch_data, input_letter, state[0]))
                # logging.info("original input is {}\tletter I wanna have is '{}'\tthe state is {}".format(batch_data, input_letter, state[0]))
                f.write(state[0]+'\n')
                f.flush()

if __name__ == '__main__':
    args=get_args()
    output_dir = 'results/'+args.ckpt_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train(args,output_dir)
