import argparse

def get_args():

    parser = argparse.ArgumentParser(description="model parameters")
    parser.add_argument('--output_dir', type=str, default="toy/", help='Output directory path to store checkpoints.')
    parser.add_argument('--gen_path', type=str, default="../output.txt", help='Output data filepath for predictions on test data.')

    ## Model building
    parser.add_argument('--max_len', type=int, default=16,help='Input length of model')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    parser.add_argument('--max_key', default=10, type=float)
    parser.add_argument('--max_key_rate', default=0.5, type=float)
    parser.add_argument('--style_mode', default='plm', type=str,help='plm | pipeline | textcnn')
    parser.add_argument('--plm_name',default='../EleutherAI/gpt-j-6B',type=str)
    parser.add_argument('--topk', default=5, type=int,help="top-k words in masked out word prediction")
    parser.add_argument("--direction", type=str, default='0-1',help='0-1 | 1-0')
    parser.add_argument("--fluency_weight", type=int, default=2, help='fluency')
    parser.add_argument("--sem_weight",type=int, default=3, help='semantic similarity')
    parser.add_argument("--style_weight", type=int, default=12, help='style')
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--buffer_size", type=int, default=200)
    parser.add_argument("--num_actions", type=int, default=3)
    # parser.add_argument('--fp32', action="store_true")
    parser.add_argument('--fp32', type=bool, default=True)
    parser.add_argument("--tau", type=float, default=1e-3,help='for soft update of target parameters')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--dst', default='yelp', type=str,help='yelp | gyafc | amazon')
    parser.add_argument("--setting", type=str, default='zero-shot')
    parser.add_argument("--delim", type=str, default='curly_bracket')
    parser.add_argument("--prompt_setting", type=str, default='classification_v2')
    parser.add_argument("--bsz",type=int,default=1,help="batch size")
    parser.add_argument("--path",type=str,default='toy',help="path for checkpoints")
    parser.add_argument("--gamma", type=int, default=0.99, help="discount factor")
    parser.add_argument("--update_interval", type=int, default=200, help="update_interval")
    parser.add_argument("--ckpt_num", type=int, default=1400, help="update_interval")
    # parser.add_argument("--world_size", type=int, default=2, help="multiple gpu processing")
    # parser.add_argument("--mode", type=str, default='train')

    ## Ablation Study:
    parser.add_argument("--semantic_mode", default='kw-sent',type=str,help='kw | sent | kw-sent')
    parser.add_argument("--action",default='all', type=str, help='replace | delete | insert | all')
    parser.add_argument('--keyword_pos', default=True, type=bool)
    parser.add_argument("--early_stop",default=False, type=bool)
    parser.add_argument("--same_pos_edit", default=False, type=bool)
    parser.add_argument("--ablation",default=None,type=str, help =" flu | sem | style")

    args = parser.parse_args()

    return args


