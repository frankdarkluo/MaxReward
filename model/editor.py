import torch
import torch.nn as nn
import numpy as np
import RAKE
import nltk
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import logging
logging.set_verbosity_error()
special_tokens = ['_num_', "'s","'d","'m","'re","'ve","'ll","n't"]

class RobertaEditor(nn.Module):
    def __init__(self, opt,device,rbt_model, rbt_tokenizer):
        super(RobertaEditor, self).__init__()
        self.opt = opt
        self.topk = opt.topk
        self.model=rbt_model
        self.tokenizer=rbt_tokenizer
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.ops_map = [self.replace, self.insert, self.delete]
        self.max_len = opt.max_len
        self.Rake = RAKE.Rake(RAKE.SmartStopList())
        self.device=device

        print("Editor built")

    def edit(self, inputs, ops, positions):
        # mask_inputs=[]
        mask_inputs=[self.ops_map[op](inp, position) if position < len(inp.split()) else "" for inp, op, position in zip(inputs, ops, positions)]
        if ops[0] < 2:  # replacement or insertion, have a mask
            index = [idx for idx, masked_input in enumerate(mask_inputs) if '<mask>' in masked_input]
            mask_inputs = np.array(mask_inputs)[index]
            mask_outputs=self.generate(mask_inputs.tolist())[0]
            return mask_outputs
        else:
            return mask_inputs

    def generate(self, sents):

        inputs = self.tokenizer.batch_encode_plus(sents, return_tensors="pt", padding=True,
                                                  max_length=32, truncation=True)
        mask_indices = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        # Send input tensors to device (GPU if available)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        mask_word_predictions = logits[torch.arange(len(sents)), mask_indices].topk(self.topk, dim=-1).indices.squeeze(-1)
        mask_words_list = [self.tokenizer.batch_decode(pred) for pred in mask_word_predictions]


        # process_mask_words_list=[]
        # # delete the stopwords
        # for token in mask_words_list:
        #     token=token.lower()
        #     # token has no overlap in stopwords
        #     if token.isdigit() == True:
        #         _token = [token]
        #     else:
        #         _token = token
        #
        #     if len(set(_token) & set(stopwords)) == 0 and token not in 'bcdefghjklmnopqrstvwxyz' and token not in process_mask_words_list:
        #         process_mask_words_list.append(token)

        # filled_sentences = [[sent.replace(tokenizer.mask_token, mask_word.strip().lower()) for mask_word in mask_words] for
        #                     sent, mask_words in zip(sentences, mask_word_lists)]
        filled_sentences = []
        for sent, mask_words in zip(sents, mask_words_list):
            sentences = []
            for mask_word in mask_words:
                sentences.append(sent.replace(self.tokenizer.mask_token, mask_word.strip().lower()))
            filled_sentences.append(list(set(sentences)))

        return filled_sentences

    def insert(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + ["<mask>"] + input_texts.split()[mask_idx:]
        return " ".join(input_texts_with_mask_list)

    def replace(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + ["<mask>"] + input_texts.split()[mask_idx + 1:]
        return " ".join(input_texts_with_mask_list)

    def delete(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + input_texts.split()[mask_idx + 1:]
        return " ".join(input_texts_with_mask_list)

    def state_vec(self, inputs):
        sta_vec_list = []
        pos_list = []

        for line in inputs:
            line = ' '.join(line.split()[:self.max_len])

            sta_vec = list(np.zeros([self.max_len]))
            keyword = self.Rake.run(line)
            pos_tags = nltk.tag.pos_tag(line.split())
            pos = [x[1] for x in pos_tags]
            pos_list.append(pos)

            if keyword != []:
                keyword=list(list(zip(*keyword))[0])
                keyword_new = []
                linewords = line.split()
                for i in range(len(linewords)):
                    for item in keyword:
                        length11 = len(item.split())
                        if ' '.join(linewords[i:i + length11]) == item:
                            keyword_new.extend([i + k for k in range(length11)])
                for i in range(len(keyword_new)):
                    ind = keyword_new[i]
                    if ind <= self.max_len - 2:
                        sta_vec[ind] = 1

            if self.opt.keyword_pos == True:
                sta_vec_list.append(self.keyword_pos2sta_vec(sta_vec, pos))
            else:
                if np.sum(sta_vec) == 0:
                    sta_vec[0] = 1
                sta_vec_list.append(sta_vec)


        return sta_vec_list,pos_list

    def mean_pooling(self, model_output, attention_mask):
        #reference: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def keyword_pos2sta_vec(self, keyword, pos):
        key_ind = []
        pos = pos[:self.max_len]
        for i in range(len(pos)):
            if keyword[i] == 1:
                key_ind.append(i)
            elif pos[i] in ['JJS', 'JJR', 'JJ', 'RBR', 'RBS', 'RB', 'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB'] \
                    and keyword[i] == 0:
                key_ind.append(i)

        #key_ind = key_ind[:max(int(option.max_key_rate * len(pos)), option.max_key)]
        sta_vec = []
        for i in range(len(keyword)):
            if i in key_ind:
                sta_vec.append(1)
            else:
                sta_vec.append(0)

        if np.sum(sta_vec) == 0:
            sta_vec[0] = 1
        return sta_vec

    def plm_token(self,lines):
        rbt_lines=[]
        for line in lines:
            plm_line = []
            line=line.split()
            for idx, token in enumerate(line):
                if idx==0:
                    plm_line.append(token)
                else:
                    if token in ["'s","'d","'m","'re","'ve","'ll","n't","<mask>"]:
                        plm_line.append(token)
                    else:
                        token = 'Ġ' + token
                        plm_line.append(token)
            plm_line=plm_line[:self.max_len]
            rbt_line = ['<s>'] + plm_line + ['</s>']+[self.tokenizer.pad_token]*(self.max_len-2-len(plm_line))
            rbt_lines.append(rbt_line)

        return rbt_lines