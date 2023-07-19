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
os.environ["TOKENIZERS_PARALLELISM"] = "true"
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import RobertaTokenizer, RobertaForMaskedLM
rbt_model = RobertaForMaskedLM.from_pretrained('roberta-large', return_dict=True).to(device)
rbt_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
print("loading roberta ...")

def infer(args, editor, scorer, agent):
    set_seed(args.seed)

    BSZ = args.bsz

    infer_dir = 'results/' + args.path+'/'
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
                          batch_size=BSZ,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)

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
                batch_data=sorted(batch_data, key=lambda x: len(x.split()), reverse=True)
                ref_olds=batch_data
                batch_state_vec, _ = editor.state_vec(batch_data)

                state=ref_olds

                max_episode_reward = [0 for _ in range(len(ref_olds))]
                for step in range(args.max_steps):
                    # infer actions
                    with torch.no_grad():
                        q_values = target_net(agent.text2emb(state))
                        actions = q_values.max(1)[1]
                        print("the q values for each action are",q_values)
                        logging.info("the action is {}".format(actions.item()))

                    ref_news = []
                    for idx in range(BSZ):
                        state_words = state[idx].split()
                        action=actions[idx]
                        intermediate_results = []
                        for positions in range(len(state_words)):
                            edited_state = editor.edit([state[idx]], [action], [positions])
                            intermediate_results+=edited_state

                        ref_news.append(intermediate_results)

                    # get reward
                    results = [scorer.scoring(ref_news[i], [ref_olds[i]], [batch_state_vec[i]])
                                                                        for i in range(len(ref_news))]

                    index, ref_new_score, new_style_labels = zip(*results)

                    temp_next_state = [ref_news[i][index[i]] for i in range(len(ref_news))]
                    reward=list(ref_new_score)

                    # update replay buffer
                    # if ref_new_score>ref_old_score and reward> max_episode_reward:
                    for i in range(len(ref_news)):
                        if reward[i]> max_episode_reward[i]:
                            max_episode_reward[i] = reward[i]
                            state[i] = temp_next_state[i]

                #update output.txt
                for i in range(BSZ):
                    f.write(state[i]+'\n')
                    f.flush()


def main():
    args = get_args()
    editor = RobertaEditor(args, device, rbt_model, rbt_tokenizer).to(device)
    scorer = Scorer(args, editor, device).to(device)
    agent = Agent(args, device, rbt_model, rbt_tokenizer).to(device)
    infer(args, editor, scorer, agent)

if __name__ == '__main__':
    main()
