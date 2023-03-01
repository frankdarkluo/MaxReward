import sys
sys.path.append("")
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering, \
    GPTNeoForCausalLM, GPT2Tokenizer,GPTJForCausalLM
import torch
from torch import nn
import numpy as np
from model_utils.functions import softmax

topk=1000
model_name="../EleutherAI/gpt-neo-2.7B"


model =GPTNeoForCausalLM.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#model =GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").cuda()
# model=GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",revision = "float16",
#                                       torch_dtype = torch.float16,low_cpu_mem_usage = True).cuda()
prefix="The formality of the text {"
line="the sausage and gravy were good ."
line1="and so what if it is a rebound relationship for both of you ?"
input=line1.strip()
postfix = "} is: "
# prefix = """
#     Sentence: i do not intend to be mean
#     formality: {formal}
#
#     #####
#
#     Sentence: ohhh i don't intend to be mean ..
#     formality: {informal}
#
#     #####
#
#     Sentence: what 're u doing here ?
#     formality: {informal}
#
#     #####
#
#     """
# postfix="formality: {"
# text="and so what if it is a rebound relationship for both of you ?"
# input=text = "Sentence: " + text + "\n"
seq=prefix+input+postfix

indexed_tokens = tokenizer.encode(seq)
# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens]).cuda()
# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# Get the predicted next sub-word
# if [0, -1, :] --> dim_size (1, 50257); if [:, -1, :] --> (50257,)
probs = predictions[0, -1, :]

# pos_logits=probs[tokenizer.encode('positive')]
# neg_logits=probs[tokenizer.encode('negative')]
pos_logits=probs[tokenizer.encode(' formal')]
neg_logits=probs[tokenizer.encode(' informal')]
emo_logits=torch.concat([pos_logits,neg_logits])
softmax_emo_logits=softmax(emo_logits)

pos_prob=softmax_emo_logits[0]
neg_prob=softmax_emo_logits[1]



