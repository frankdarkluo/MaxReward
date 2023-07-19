import os
from model.Scorer import Scorer
from model.editor import RobertaEditor
from model.config import get_args
from model.DQNSearchAgent import Agent, DQN
import warnings
from model.nwp import set_seed
import logging
from dateutil import tz
import torch
from toy import edit,get_reward
import string
import random
os.environ["TOKENIZERS_PARALLELISM"] = "true"
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import RobertaTokenizer, RobertaForMaskedLM
rbt_model = RobertaForMaskedLM.from_pretrained('roberta-large', return_dict=True).to(device)
rbt_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

print("loading roberta ...")

def load_data(args):
    infer_dir = 'results/' + args.path+'/'
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)

    infer_file = infer_dir+'inference.txt'
    with open('data/toy_test.txt', 'r', encoding='utf8') as f:
        data = f.readlines()
        test_data = [line.strip() for line in data]

    log_txt_path=os.path.join(infer_file.split('.txt')[0] + '.log')
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='',filename=log_txt_path,filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logging.info(args)

    return infer_dir,infer_file,test_data

# Function to create reference news given a state and actions
def create_ref_news(state, action, MAX_LEN):
    ref_news = []
    cur_state = state[0][:MAX_LEN]

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

def infer_action(target_net, agent, state):
    with torch.no_grad():
        q_values = target_net(agent.text2emb(state))

        action = q_values.max(1)[1]

        while len(state[0]) == 1 and action.item() == 2:
            action = torch.tensor(random.choice([0,1])).to(device)

    print("the q values for each action are", q_values)
    logging.info("the action is {}".format(action.item()))

    return action

def infer(args, agent):
    BSZ = args.bsz
    MAX_LEN=10

    infer_dir,infer_file,test_data=load_data(args)

    # start inference
    print("start inference...")
    with open(infer_file, 'w', encoding='utf8') as f:

        # load target model
        ckpt_path=os.path.join(infer_dir+str(args.ckpt_num)+'_target_net.pt')
        target_net = DQN(agent.state_dim, args.num_actions).to(device)
        target_net.load_state_dict(torch.load(ckpt_path))
        target_net.eval()

        # batch inference
        with torch.no_grad():
            for idx,batch_data in enumerate(test_data):
                if idx != 0 and idx % 20 == 0:
                    print("inference batch: {}".format(idx))
                ref_olds=batch_data

                # generate a random input letter
                input_letter = random.choice(string.ascii_lowercase)
                state=[ref_olds]

                max_episode_reward = float('-inf')
                for step in range(args.max_steps):

                    # infer actions
                    action=infer_action(target_net, agent, state)

                    ref_news=create_ref_news(state, action, MAX_LEN)

                    # get max reward
                    reward, temp_next_state = get_reward(ref_news, input_letter)

                    cur_state=state
                    accept = False

                    # update replay buffer
                    for i in range(BSZ):
                        if reward > max_episode_reward:
                            max_episode_reward = reward
                            state = [temp_next_state]
                            accept=True
                        else: state = state

                    print(
                        "reward is {:.6f}, old_state is {}\taction is {},\tletter I wanna have is '{}'\t"
                        "the new state is {}\t accept is {}"
                        .format(reward, cur_state[0], str(action.item()), input_letter, state[0], accept))

                    logging.info(
                        "reward is {:.6f}, old_state is {}\taction is {},\tletter I wanna have is '{}'\t"
                        "the new state is {}\t accept is {}"
                            .format(reward, cur_state[0], str(action.item()), input_letter, state[0], accept))


                #update output.txt
                for i in range(BSZ):
                    f.write(state[i]+'\n')
                    print(f"the final output for {ref_olds} is {state[i]}\n")
                    logging.info(f"the final output for {ref_olds} is {state[i]}\n")
                    f.flush()


def main():
    args = get_args()
    agent = Agent(args, device, rbt_model, rbt_tokenizer).to(device)
    infer(args, agent)

if __name__ == '__main__':
    main()
