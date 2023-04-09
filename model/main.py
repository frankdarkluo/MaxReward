import math
import os
import logging
from DQNSearchAgent import Agent
from Scorer import Scorer
from editor import RobertaEditor
from config import get_args
import torch.multiprocessing as mp
import warnings
from model.nwp import set_seed
import datetime
from dateutil import tz
import torch
from torch.utils.data import DataLoader
from utils.helper import plot
from utils.dataset import TSTDataset
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time

try:
    mp.set_start_method('spawn', force=True)
    print("spawned")
except RuntimeError:
    pass

def main():
    args = get_args()
    set_seed(args.seed)
    editor = RobertaEditor(args).to(device)
    scorer= Scorer(args, editor).to(device)
    agent=Agent(editor,args).to(device)

    BSZ = args.bsz
    dst=args.dst

    of_dir = '../results/' + args.output_dir
    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

    # log_txt_path=os.path.join(of_dir, output_file.split('.txt')[0] + '.log')
    # print(log_txt_path)
    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)
    # logging.basicConfig(format='',filename=log_txt_path,filemode='w',
    #                     datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    # logging.info(args)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    train_dataset=TSTDataset(args,'process_train')
    train_data=DataLoader(train_dataset,
                          batch_size=BSZ,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)
    # val_data=TSTDataset(data)
    # test_data=TSTDataset(data)


    # training
    with mp.Pool(processes=4) as pool:
        for idx,batch_data in enumerate(train_data):
            batch_data=sorted(batch_data, key=lambda x: len(x.split()), reverse=True)
            ref_oris=ref_olds=batch_data
            batch_state_vec, _ = editor.state_vec(batch_data)

            # epsilon-exploration for choosing an action
            all_rewards = []
            losses = []

            state=ref_olds

            # training Q-network
            start_time = time.time()
            max_episode_reward = [scorer.acceptance_prob(ref_olds[i], [ref_oris[i]], [batch_state_vec[i]])[1] for i in range(len(ref_olds))]
            for step in range(args.max_steps):

                torch.cuda.empty_cache()

                epsilon = epsilon_by_frame(idx)
                action = agent.act(state, epsilon)

                # ref_news = [editor.edit(state, [action] * BSZ, [positions] * BSZ) for positions in range(max_seq_len)]
                # ref_news=[sum([editor.edit([state[idx]], [action], [positions])[0] for positions in range(len(state[idx].split()))],[]) for idx in range(BSZ)]
                ref_news = []
                for idx in range(BSZ):
                    state_words = state[idx].split()
                    intermediate_results = []
                    for positions in range(len(state_words)):
                        edited_state = editor.edit([state[idx]], [action], [positions])
                        intermediate_results+=edited_state

                    ref_news.append(intermediate_results)


                if step<args.max_steps-1:
                    done=False
                else: done=True # meaning when step=4, done=True

                next_state=state

                # get reward
                results = [scorer.acceptance_prob(ref_news[i], [ref_olds[i]], [batch_state_vec[i]])
                                                                    for i in range(len(ref_news))]

                end_time = time.time()
                elapsed_time = end_time - start_time
                print('代码执行时间：{:.6f}秒'.format(elapsed_time))

                index, ref_new_score, new_style_labels = zip(*results)

                temp_next_state = [ref_news[i][index[i]] for i in range(len(ref_news))]
                reward=list(ref_new_score)

                # update replay buffer
                # if ref_new_score>ref_old_score and reward> max_episode_reward:
                for i in range(len(ref_news)):
                    if reward[i]> max_episode_reward[i]:
                        max_episode_reward[i] = reward[i]
                        next_state[i] = temp_next_state[i]

                #update replay buffer
                for i in range(BSZ):
                    agent.replay_buffer.push(state[i], action, max_episode_reward[i], next_state[i], done)

                    # update state
                    state[i]=next_state[i]

                # update Q-network
                if len(agent.replay_buffer) >= args.buffer_size:
                    loss = agent.compute_td_loss()
                    print("loss is {}".format(str(loss.item())))
                    losses.append(loss.item())

                if done:
                    all_rewards.append(max_episode_reward)

            if idx % 100 == 0:
                plot(idx, all_rewards, losses)


if __name__ == '__main__':
    main()
