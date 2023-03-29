import torch
import torch.nn as nn
import sys
import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.nn import CrossEntropyLoss
from transformers import logging
logging.set_verbosity_error()
from utils.constant import *
sys.path.append("")
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel,AutoTokenizer,AutoModelForCausalLM
from utils.helper import pytorch_cos_sim
from nwp import predict_next_word
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Scorer(nn.Module):
    def __init__(self, opt,editor):
        super(Scorer,self).__init__()
        self.opt = opt
        self.editor = editor
        self.flu_w = opt.fluency_weight  # 3
        self.sem_w = opt.sem_weight
        self.style_w = opt.style_weight
        self.stride = 1024
        self.abl = self.opt.ablation
        self.dst = self.opt.dst
        self.p_setting = self.opt.prompt_setting
        self.memory = []

        if self.opt.style_mode == 'plm':
            if 'gpt-j-hf' in self.opt.plm_name:
                self.plm = GPTNeoForCausalLM.from_pretrained(self.opt.plm_name, revision="float16")
                self.plm.half()
            else:
                self.plm = AutoModelForCausalLM.from_pretrained(self.opt.plm_name, revision="float16")
            self.plm.eval()
            self.plm.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = self.opt.max_len
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.ppl_max_len = self.model.config.n_positions

    def style_scorer(self,ref_news):

        if self.opt.setting == 'zero-shot':
            num_es = 0
            delim_left1,delim_right1="",""
        else:
            num_es = 4
            delim_left1, delim_right1="{","}"

        prefix = create_exemplars(self.dst,num_es, self.p_setting, delim_left1, delim_right1)
        prompts= [write_sentence(self.dst, self.p_setting, "{", "}", text) for text in ref_news]
        input_candidate_text = [prefix + prompt for prompt in prompts]

        style_probs, style_labels = predict_next_word(self.plm, self.tokenizer, input_candidate_text,
                                                        direction=self.opt.direction)
        prob_new_probs=torch.pow(style_probs, self.style_w)

        return prob_new_probs,style_labels

    def fluency_scorer(self,ref_news): #ref: https://huggingface.co/docs/transformers/perplexity

        encodings = self.tokenizer(ref_news, return_tensors="pt",padding=True,max_length=self.max_len).to(device)
        input_ids = encodings.input_ids
        begin_loc = max(self.stride - self.ppl_max_len, 0)
        end_loc = min(self.stride, input_ids.size(1))
        trg_len = end_loc  # may be different from stride on last loop
        input_ids =input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        # Ref: https://github.com/huggingface/transformers/issues/473
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            lm_logits=outputs[1]
            shift_logits = lm_logits[..., :input_ids.shape[1]-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct=CrossEntropyLoss(ignore_index=-100,reduction='none')
            loss=loss_fct(shift_logits.view(-1,lm_logits.size(-1)),shift_labels.view(-1))
            ppl = torch.exp((loss.reshape(-1, input_ids.shape[1]-1)).mean(dim=1))

        return 1/ppl.pow(self.flu_w)

    def keyword_sim(self,ref_new_embeds,ref_old_embeds,state_vec=None):
        e = 1e-5
        repeat_num=ref_new_embeds.size(0)
        emb1 = ref_new_embeds.permute(0, 2, 1)
        emb2 = ref_old_embeds.repeat(repeat_num,1,1)
        emb_mat = torch.bmm(emb2, emb1)
        state_vec= torch.tensor(state_vec, dtype=torch.bool)
        weight2 = state_vec.repeat(repeat_num,state_vec.shape[1])[:,:emb2.shape[1]]
        norm2 = 1 / (torch.norm(emb2, p=2, dim=2) + e)  # K,8,8
        norm1 = 1 / (torch.norm(emb1, p=2, dim=1) + e)  # K,7,7
        diag_norm2 = torch.diag_embed(norm2)  # 2D-->3D
        diag_norm1 = torch.diag_embed(norm1)
        sim_mat = torch.bmm(torch.bmm(diag_norm2, emb_mat), diag_norm1)  # K,8,7
        sim_vec, _ = torch.max(sim_mat, dim=2)  # K,8
        try:
            kw_similarity, _ = torch.min(sim_vec[weight2].reshape(repeat_num,-1), dim=1)
        except:
            weight2[:,0]=True
            kw_similarity, _ = torch.min(sim_vec[weight2].reshape(repeat_num,-1), dim=1)
        return kw_similarity

    def semantic_scorer(self,ref_news, ref_olds,state_vec=None):

        ref_new_embeds, mean_new_embeds = self.editor.get_contextual_word_embeddings(ref_news)
        ref_old_embeds, mean_old_embeds = self.editor.get_contextual_word_embeddings(ref_olds)

        #-----keyword-level sim------
        if self.opt.semantic_mode=='kw':
            kw_sim=self.keyword_sim(ref_new_embeds,ref_old_embeds,state_vec)
            similarity = kw_sim.pow(self.sem_w)

        #-----sent-level sim------
        elif self.opt.semantic_mode=='sent':
            sent_sim=pytorch_cos_sim(mean_new_embeds, mean_old_embeds)
            similarity=sent_sim.pow(self.sem_w)

        # -----kw-sent level sim------
        elif self.opt.semantic_mode=='kw-sent':
            kw_sim=self.keyword_sim(ref_new_embeds,ref_old_embeds,state_vec)
            sent_sim= pytorch_cos_sim(mean_new_embeds, mean_old_embeds)
            similarity = kw_sim.pow(self.sem_w)* sent_sim.pow(self.sem_w).squeeze()

        return similarity

    def scorer(self, input_news,ref_oris,state_vec=None):
        fluency_scores=self.fluency_scorer(input_news) # input-news, ref_oris-->["I like you"]
        style_scores,style_labels=self.style_scorer(input_news) # input-news, ref_oris-->["I like you"]
        sim_scores = self.semantic_scorer(input_news, ref_oris, state_vec).squeeze()

        if self.abl=='sty':
            total_scores = fluency_scores* sim_scores
        elif self.abl=='fl':
            total_scores = sim_scores * style_scores
        elif self.abl=='sem':
            total_scores = fluency_scores * style_scores
        else:# None
            total_scores = fluency_scores * sim_scores * style_scores

        return total_scores.squeeze(),style_scores.squeeze(), style_labels

    def acceptance_prob(self, input_news, input_olds,ref_oris,state_vec):
        ref_old_score, old_style_score, _ = self.scorer(input_olds, ref_oris, state_vec)
        ref_new_scores,new_style_scores,new_style_labels=self.scorer(input_news,ref_oris,state_vec)

        ref_new_score_index=torch.argmax(ref_new_scores)
        ref_new_score=torch.max(ref_new_scores)

        seq_old=len(input_olds[0].split())
        seq_new=len(input_news[0].split())

        # V_old=torch.log(torch.maximum(torch.pow(ref_old_score, 1.0 / seq_old), torch.tensor(1e-200))).item()
        # V_new=torch.log(torch.maximum(torch.pow(ref_new_score, 1.0 / seq_new), torch.tensor(1e-200))).item()
        V_old = np.log(np.maximum(np.power(ref_old_score.cpu().detach().numpy(), 1.0 / seq_old), 1e-200))
        V_new = np.log(np.maximum(np.power(ref_new_score.cpu().detach().numpy(), 1.0 / seq_old), 1e-200))

        # Scale the score to be [-3,3]
        V_old, V_new = V_old * 3, V_new * 3

        if V_new - V_old > 0:
            accept_hat = [1]
        else:
            accept_hat = [0]

        return ref_new_score_index,V_old,V_new,new_style_labels, accept_hat