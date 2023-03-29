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
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time

start_time=time.time()
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

    MAX_LEN=args.max_len
    BSZ = args.bsz
    dst=args.dst

    of_dir = '../results/' + args.output_dir
    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

    output_file ='{}_{}_seed={}_{}_{}_{}.txt'.\
        format(timestamp,dst,str(args.seed),args.style_mode,str(args.style_weight),args.direction)
    log_txt_path=os.path.join(of_dir, output_file.split('.txt')[0] + '.log')
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='',filename=log_txt_path,filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logging.info(args)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    train_dataset=TSTDataset(args,'train')
    train_data=DataLoader(train_dataset,
                          batch_size=BSZ,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)
    # val_data=TSTDataset(data)
    # test_data=TSTDataset(data)

    end_time=time.time()
    elapsed_time = end_time - start_time

    print('代码执行时间：{:.6f}秒'.format(elapsed_time))

    # training
    with open(of_dir + output_file, 'w', encoding='utf8') as f, mp.Pool(processes=4) as pool:
        for idx,batch_data in enumerate(train_data):

            ref_oris=ref_olds=batch_data
            batch_state_vec, _ = editor.state_vec(batch_data)

            seq_len=[len(line.split()) for line in batch_data]
            max_seq_len=max(seq_len)

            # epsilon-exploration for choosing an action
            all_rewards = []
            losses = []


            state=ref_olds

            # training Q-network
            for step in range(args.max_steps):

                torch.cuda.empty_cache()

                epsilon = epsilon_by_frame(idx)
                action = agent.act(state, epsilon)

                ref_news = pool.starmap(editor.edit, [(state, [action] * BSZ, [positions] * BSZ, BSZ, MAX_LEN) for positions in range(max_seq_len)])
                if step<args.max_steps-1:
                    done=False
                else: done=True # meaning when step=4, done=True

                max_episode_reward = -1 * float("inf")
                next_state=None

                # get reward
                for idx in range(len(ref_news)):
                    ref_new_batch_data = ref_news[idx]

                    results = pool.starmap_async(scorer.acceptance_prob,
                                           [(ref_new_batch_data[i], [ref_olds[i]], [ref_oris[i]], [batch_state_vec[i]])
                                            for i in range(len(ref_new_batch_data))])
                    results = results.get()
                    index, ref_old_score, ref_new_score, new_style_labels, _ = zip(*results)
                    # index, ref_old_score, ref_new_score, new_style_labels, _ \
                    #     = scorer.acceptance_prob(ref_new_batch_data, ref_olds, ref_oris, state_vec)


                    temp_next_state = [ref_new_batch_data[i][index[i]] for i in range(BSZ)]
                    reward=list(ref_new_score)

                    # update replay buffer
                    # if ref_new_score>ref_old_score and reward> max_episode_reward:
                    for i in range(BSZ):
                        if reward[i]> max_episode_reward[i]:
                            max_episode_reward[i] = reward[i]
                            next_state[i] = temp_next_state[i]


                #update replay buffer
                for i in range(BSZ):
                    agent.replay_buffer.push(state[i], action[i], max_episode_reward[i], next_state[i], done)

                # update state
                state[i]=next_state[i]
                print("the best candidate in step {} is {}".format(step,state))


                # update Q-network
                if len(agent.replay_buffer) >= args.buffer_size:
                    loss = agent.compute_td_loss()
                    print("loss is {}".format(str(loss.item())))
                    losses.append(loss.item())

                if done:
                    all_rewards.append(max_episode_reward)
                    logging.info("the generated candidate is {}: ".format(state)) # update log
                    f.write(state) # update the .txt
                    f.flush()
                    max_episode_reward = -1 * float("inf") # refresh

            if idx % 100 == 0:
                plot(idx, all_rewards, losses)


if __name__ == '__main__':
    main()
