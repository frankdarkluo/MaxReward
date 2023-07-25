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
from torch.utils.data import DataLoader
from utils.dataset import TSTDataset
from main import create_input_news, get_reward

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import RobertaTokenizer, RobertaForMaskedLM
rbt_model = RobertaForMaskedLM.from_pretrained('roberta-large', return_dict=True).to(device)
rbt_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
print("loading roberta ...")


def load_data(args):
    
    infer_dir = 'results/' + args.ckpt_path+'/'
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)

    infer_file = infer_dir+'inference.txt'
    log_txt_path=infer_file.split('.txt')[0] + '.log'
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='',filename=log_txt_path,filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logging.info(args)

    test_dataset=TSTDataset(args,'test')
    test_data=DataLoader(test_dataset,
                          batch_size=args.bsz,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)
    
    return test_data,infer_dir,infer_file

def infer(args, editor, scorer, agent):

    BSZ = args.bsz

    test_data, infer_dir, infer_file=load_data(args)

    # start inference
    print("start inference...")
    with open(infer_file, 'w', encoding='utf8') as f:

        # load target model
        # agent.state_dim=50265
        ckpt_path=os.path.join(infer_dir+str(args.ckpt_num)+'_target_net.pt')
        target_net = DQN(agent.state_dim, args.num_actions).to(device)
        target_net.load_state_dict(torch.load(ckpt_path))
        target_net.eval()

        # batch inference
        with torch.no_grad():
            for idx,batch_data in enumerate(test_data):
                if idx != 0 and idx % 20 == 0:
                    print("inference batch: {}".format(idx))
                batch_data=sorted(batch_data, key=lambda x: len(x.split()), reverse=True)
                input_olds=batch_data
                batch_state_vec, _ = editor.state_vec(batch_data)

                state=input_olds

                max_episode_reward = [0 for _ in range(len(input_olds))]
                for step in range(args.max_steps):
                    # infer actions
                    with torch.no_grad():
                        q_values = target_net(agent.text2emb(state))
                        actions = q_values.max(1)[1]
                        print("the q values for each action are",q_values)
                        logging.info("the action is {}".format(actions.item()))

                    input_news = create_input_news(args.bsz, state, actions, editor)

                    # get editing results
                    results = [scorer.scoring(input_news[i], [input_olds[i]], [batch_state_vec[i]])
                                                                        for i in range(len(input_news))]

                    # get reward
                    reward, best_cand_states = get_reward(results, input_news)

                    for i in range(len(input_news)):
                        if reward[i]> max_episode_reward[i]:
                            max_episode_reward[i] = reward[i]
                            state[i] = best_cand_states[i]

                #update output.txt
                for i in range(BSZ):
                    f.write(state[i]+'\n')
                    f.flush()

def main():
    args = get_args()
    set_seed(args.seed)
    editor = RobertaEditor(args, device, rbt_model, rbt_tokenizer).to(device)
    scorer = Scorer(args, editor, device).to(device)
    agent = Agent(args, device, rbt_model, rbt_tokenizer).to(device)
    infer(args, editor, scorer, agent)

if __name__ == '__main__':
    main()
