# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append("../")
import model_args
args=model_args.get_args()
import torch
from torch import cuda
from transformers import RobertaTokenizer
import string
from model_utils.dataset import SCIterator
from model_utils.textcnn import TextCNN
from editor import RobertaEditor
editor=RobertaEditor(args)

filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]
device = 'cuda' if cuda.is_available() else 'cpu'
special_tokens = [{'bos_token': '<bos>'},
                  {'eos_token': '<eos>'}, {'sep_token': '<sep>'},
                  {'pad_token': '<pad>'}, {'unk_token': '<unk>'}]
#tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    parser = argparse.ArgumentParser('Evaluating Style Strength')
    parser.add_argument('--gen_path', default='../output.txt', type=str, help='src')
    parser.add_argument('--direction', default='0-1', type=str, help='from 0 to 1')

    parser.add_argument('--max_len', default=30, type=int, help='max tokens in a batch')
    parser.add_argument('--embed_dim', default=300, type=int, help='the embedding size')
    parser.add_argument('--dataset', default='gyafc', type=str, help='the name of dataset')
    parser.add_argument('--model', default='textcnn', type=str, help='the name of model')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('--batch_size', default=1, type=int, help='max sents in a batch')
    parser.add_argument("--dropout", default=0.5, type=float, help="Keep prob in dropout")

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    test_src, test_tgt = [], []
    with open(opt.gen_path,'r') as f:
        for line in f.readlines():
            line = line.strip().lower()
            # line=line.split()
            # line = "".join(
            #     [" " + i if not i.startswith("'") and i not in ".," else i for i in line]).strip()
            input_seq=tokenizer.encode(line.strip())[:opt.max_len]

            test_src.append(input_seq)
    max_num=len(test_src)
    print('[Info] {} instances from src test set'.format(len(test_src)))
    test_loader = SCIterator(test_src, test_tgt, opt, tokenizer.pad_token_id)

    model = TextCNN(opt.embed_dim, len(tokenizer), filter_sizes,
                    num_filters, None, dropout=opt.dropout)
    model.to(device).eval()
    model.load_state_dict(torch.load('../checkpoints/textcnn_{}_roberta.chkpt'.format(
        opt.dataset)))

    total_num = max_num
    total_acc = 0.
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            x_batch, _ = map(lambda x: x.to(device), batch)
            logits = model(x_batch)
            #print(F.softmax(logits, dim=-1))
            _, y_hat = torch.max(logits,dim=-1)

            if opt.direction=='0-1' and y_hat==1:
                print(i)
                total_acc+=1

            elif opt.direction=='1-0' and y_hat==0:
                print(i)

                total_acc+=1

    print(total_acc)
    print(total_num)

    print('Test: {}'.format('acc {:.4f}%').format(
        total_acc / total_num * 100))


if __name__ == '__main__':
    main()