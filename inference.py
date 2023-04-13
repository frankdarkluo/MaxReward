import os
from model.DQNSearchAgent import Agent
from model.Scorer import Scorer
from model.editor import RobertaEditor
from model.config import get_args
import torch.multiprocessing as mp
import warnings
from model.nwp import set_seed
import datetime
from dateutil import tz
import torch
from torch.utils.data import DataLoader
from utils.dataset import TSTDataset
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer(args, editor, scorer, agent):
    set_seed(args.seed)

    BSZ = args.bsz

    of_dir = 'results/' + args.output_dir
    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

    infer_file = '{}_output.txt'.format(timestamp)

    test_dataset=TSTDataset(args,'test')
    test_data=DataLoader(test_dataset,
                          batch_size=BSZ,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)

    # start inference
    print("start inference...")
    with open(of_dir + infer_file, 'w', encoding='utf8') as f:

        # load model
        target_net=agent.load_model(args.path)
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

                    torch.cuda.empty_cache()

                    # infer actions
                    with torch.no_grad():
                        q_values = target_net(agent.text2emb(state))
                        actions = q_values.max(1)[1]

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
                    results = [scorer.acceptance_prob(ref_news[i], [ref_olds[i]], [batch_state_vec[i]])
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
    editor = RobertaEditor(args).to(device)
    scorer = Scorer(args, editor).to(device)
    agent = Agent(editor, args).to(device)
    infer(args, editor, scorer, agent)

if __name__ == '__main__':
    main()
