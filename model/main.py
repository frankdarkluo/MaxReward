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

from utils.helper import plot
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    if args.direction == '0-1': postfix = '0'
    else: postfix = '1'

    filename='../data/{}/test.{}'.format(dst,postfix)
    with open(filename, 'r', encoding='utf8') as f:
        data = f.readlines()[:]

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

    output_file ='{}_{}_seed={}_{}_{}_{}.txt'.\
        format(timestamp,dst,str(args.seed),args.style_mode,str(args.style_weight),args.direction)
    log_txt_path=os.path.join(of_dir, output_file.split('.txt')[0] + '.log')
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='',filename=log_txt_path,filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)

    word_pairs ={"ca n't": "can not", "wo n't": "will not"}
    logging.info(args)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    # training
    with open(of_dir + output_file, 'w', encoding='utf8') as f, mp.Pool(processes=4) as pool:
        for idx in range(len(data)):
            print("\nidx:", idx)
            print("------------------------------------")
            sent_data=data[BSZ * idx:BSZ * (idx + 1)]

            #preprocessing
            ref_oris = []
            for d in sent_data:
                for k, v in word_pairs.items():
                     d=d.strip().lower().replace(k, v)
                ref_oris.append(d)

            ref_olds=ref_oris.copy()

            state_vec, _ = editor.state_vec(ref_olds)
            seq_len=[len(line.split()) for line in ref_olds]
            max_seq_len=max(seq_len)

            # epsilon-exploration for choosing an action
            all_rewards = []
            losses = []

            # BSZ=1
            state=ref_olds[0]

            # training Q-network
            for step in range(args.max_steps):

                torch.cuda.empty_cache()

                epsilon = epsilon_by_frame(idx)
                action = agent.act(state, epsilon)

                ref_news = pool.starmap(editor.edit, [([state], [action] * BSZ, [positions] * BSZ, BSZ, MAX_LEN)
                                                      for positions in range(max_seq_len)])
                if step<args.max_steps:
                    done=False
                else: done=True

                max_episode_reward = 0

                # get reward
                for idx in range(len(ref_news)):
                    ref_new_batch_data = ref_news[idx]
                    index, ref_old_score, ref_new_score, new_style_labels, _ \
                        = scorer.acceptance_prob(ref_new_batch_data, ref_olds, ref_oris, state_vec)

                    temp_next_state = ref_new_batch_data[index]
                    # new_style_label = new_style_labels[index]
                    reward=ref_new_score.item()/1e5

                    # update replay buffer
                    # if ref_new_score>ref_old_score and reward> max_episode_reward:
                    if reward> max_episode_reward:
                        max_episode_reward = reward
                        next_state = temp_next_state

                #update replay buffer
                agent.replay_buffer.push(state, action, max_episode_reward, next_state, done)

                # update state
                state=next_state
                print("the best candidate in step {} is {}".format(step,state))


                # update Q-network
                if len(agent.replay_buffer) >= args.buffer_size:
                    loss = agent.compute_td_loss()
                    print("loss is {}".format(str(loss.item())))
                    losses.append(loss.item())

                if done:
                    all_rewards.append(max_episode_reward)
                    logging.info("the generated candidate is {}".format(state)) # update log
                    f.write(state) # update the .txt
                    f.flush()
                    max_episode_reward = 0 # refresh

            if idx % 100 == 0:
                plot(idx, all_rewards, losses)


if __name__ == '__main__':
    main()
