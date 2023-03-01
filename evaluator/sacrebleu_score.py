import os
import sys
from nltk import wordpunct_tokenize
from sacrebleu.metrics import BLEU
import string
import argparse

def load_data(file):
    strs = []
    with open(file, 'r', encoding='utf8') as of:
        datas = of.readlines()
        for idx, data in enumerate(datas):
            data=data.strip().lower()
            strs.append(data)
    return strs

def load_ref_data(ref_path,N=50):
    refs=[[]]*N

    for file in os.listdir(ref_path):
        with open(ref_path+file,'r',encoding='utf8') as f:
            lines=f.readlines()[:N]
            for j, line in enumerate(lines):
                line = line.strip().lower()
                line = line.split()
                line = "".join(
                    [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in line]).strip()
                # line=line.split()
                # line=wordpunct_tokenize(line)

                temp=refs[j].copy()
                temp.append(line)
                refs[j]=temp.copy()
    return refs


def metric(args):
    infer =load_data(args.gen_path)
    ref_path='../data/{}/{}/'.format(args.dataset,args.task)
    golden=load_ref_data(ref_path,args.N)

    # eval bleu
    bleu = BLEU()
    r_bleu = bleu.corpus_score(infer, golden)
    print('BLEU', r_bleu)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--ref_path', default='data/gyafc/pos2neg_ref/', type=str)
    parser.add_argument('--dataset',default='yelp',type=str)
    parser.add_argument('--task', default='neg2pos_ref', type=str)
    parser.add_argument('--gen_path', default='../output.txt', type=str)
    parser.add_argument("--N",default=500,type=int)
    args = parser.parse_args()
    metric(args)
