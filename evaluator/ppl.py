import sys
sys.path.append("../")
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from transformers import GPT2LMHeadModel, GPT2Tokenizer,AutoTokenizer
import numpy as np
import torch
import string
from editor import RobertaEditor
from model_args import get_args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
max_length = model.config.n_positions
stride = 1024

args=get_args()
args.gen_path=='../output.txt'

perpl=[]
with open(args.gen_path,'r',encoding='utf8') as f:
    datas=f.readlines()
    for data in datas:
        tokens=data.strip()
        tokens=tokens.split()
        tokens="".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
        encodings = tokenizer(tokens, return_tensors="pt")
        input_ids=encodings.input_ids
        nlls = []
        for i in range(0, input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        perpl.append(ppl.cpu().numpy())

print(np.mean(perpl))