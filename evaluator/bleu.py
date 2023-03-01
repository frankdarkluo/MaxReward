import os
import sys
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk import wordpunct_tokenize

import argparse
import csv
BLEU_WEIGHTS_MEAN = [
    [1.0],
    [0.5, 0.5],
    [1/3, 1/3, 1/3],
    [0.25, 0.25, 0.25, 0.25],
]

def load_data(file):
    strs = []
    with open(file, 'r', encoding='utf8') as of:
        datas = of.readlines()#[:100]
        for idx, data in enumerate(datas):
            data=data.strip().lower()
            strs.append(data)

    str_list = [seq.split() for seq in strs]
    #str_list = [seq for seq in strs]

    return str_list

def load_ref_data(ref_path,N=50):
    refs=[[]]*N

    for file in os.listdir(ref_path):
        with open(ref_path+file,'r',encoding='utf8') as f:
            lines=f.readlines()[:N]
            for j, line in enumerate(lines):
                line = line.strip().lower()
                line=line.split()
                # line=wordpunct_tokenize(line)

                temp=refs[j].copy()
                temp.append(line.copy())
                refs[j]=temp.copy()
    return refs

def metric(args):
    infer =load_data(args.gen_path)
    ref_path='../data/{}/{}'.format(args.dataset,args.task)
    golden=load_ref_data(ref_path,args.N)

    # eval bleu
    sf=SmoothingFunction()
    corp_model_bleu1 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[0])
    corp_model_bleu2 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[1])
    corp_model_bleu3 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[2])
    corp_model_bleu4 = corpus_bleu(golden, infer, weights=BLEU_WEIGHTS_MEAN[3])

    nltk_bleu=[corp_model_bleu1, corp_model_bleu2, corp_model_bleu3, corp_model_bleu4]
    print('BLEU', nltk_bleu)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--ref_path', default='data/gyafc/pos2neg_ref/', type=str)
    parser.add_argument('--dataset',default='gyafc',type=str)
    parser.add_argument('--task', default='neg2pos', type=str)
    parser.add_argument('--gen_path', default='../data/gyafc/test.0', type=str)
    parser.add_argument("--N",default=100,type=int)
    args = parser.parse_args()
    metric(args)
