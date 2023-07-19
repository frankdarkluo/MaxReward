import torch
import torch.nn as nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.nn import CrossEntropyLoss
from transformers import logging
logging.set_verbosity_error()
from utils.constant import *
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel,AutoTokenizer
from utils.helper import pytorch_cos_sim
from model.nwp import predict_next_word


class Scorer(nn.Module):
    def __init__(self, opt,editor,device):
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
        self.device=device

        if opt.style_mode == 'plm':
            if 'gpt-j-hf' in opt.plm_name:
                self.plm = GPTNeoForCausalLM.from_pretrained(
                                                            opt.plm_name,
                                                            revision="float16",
                                                            lower_cpu_memory_usage=True)
                self.plm.half()
            else:
                self.plm = torch.load(opt.plm_name+'.pt')
            self.plm.eval()
            self.plm.to(self.device)
            print("PLM loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(opt.plm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = opt.max_len

        self.ppl_max_len = 1024

    def style_scorer(self,news):

        if self.opt.setting == 'zero-shot':
            num_es = 0
            delim_left1,delim_right1="",""
        else:
            num_es = 4
            delim_left1, delim_right1="{","}"

        prefix = create_exemplars(self.dst,num_es, self.p_setting, delim_left1, delim_right1)
        prompts= [write_sentence(self.dst, self.p_setting, "{", "}", text) for text in news]
        input_candidate_text = [prefix + prompt for prompt in prompts]

        tokens_tensor = {k: v.to(self.device) for k, v in self.tokenizer(input_candidate_text, padding=True, return_tensors="pt").items()}
        style_probs, style_labels = predict_next_word(self.plm, self.tokenizer, tokens_tensor,
                                                        direction=self.opt.direction)
        prob_new_probs=torch.pow(style_probs, self.style_w)

        return prob_new_probs,style_labels

    def fluency_scorer(self,new_embs): #ref: https://huggingface.co/docs/transformers/perplexity

        input_ids = new_embs['input_ids']
        begin_loc = max(self.stride - self.ppl_max_len, 0)
        end_loc = min(self.stride, input_ids.size(1))
        trg_len = end_loc  # may be different from stride on last loop
        input_ids =input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        # Ref: https://github.com/huggingface/transformers/issues/473
        with torch.no_grad():
            outputs = self.plm(input_ids, labels=target_ids)
            lm_logits=outputs[1]
            shift_logits = lm_logits[..., :input_ids.shape[1]-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct=CrossEntropyLoss(ignore_index=-100,reduction='none')
            loss=loss_fct(shift_logits.view(-1,lm_logits.size(-1)),shift_labels.view(-1))
            ppl = torch.exp((loss.reshape(-1, input_ids.shape[1]-1)).mean(dim=1))

        return 1/ppl.pow(self.flu_w)

    def keyword_sim(self,ref_new_embeds,ref_old_embeds,state_vec=None):
        e = 1e-5
        if ref_new_embeds.size(0)==ref_old_embeds.size(0): repeat_num=1
        else: repeat_num=ref_new_embeds.size(0)
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

    def mean_pooling(self, model_output, attention_mask):
        #reference: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_contextual_word_embeddings(self, text_embs):
        with torch.no_grad():
            outputs = self.plm(**text_embs, output_hidden_states=True)
        sentence_embeddings = self.mean_pooling(outputs, text_embs['attention_mask'])
        hidden_states = outputs.hidden_states[-1][:, 1:self.max_len+1, :].to(self.device)

        return hidden_states, sentence_embeddings

    def semantic_scorer(self,new_embs, old_embs,state_vec):

        ref_new_embeds, mean_new_embeds = self.get_contextual_word_embeddings(new_embs)
        ref_old_embeds, mean_old_embeds = self.get_contextual_word_embeddings(old_embs)

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
            sent_sim= pytorch_cos_sim(mean_new_embeds, mean_old_embeds) # seem to have some problems lol
            similarity = kw_sim.pow(self.sem_w)* sent_sim.pow(self.sem_w).squeeze()

        return similarity

    def scorer(self, fluency, sim, style):

        if self.abl=='sty':
            total_scores = fluency* sim
        elif self.abl=='fl':
            total_scores = sim * style
        elif self.abl=='sem':
            total_scores = fluency * style
        else:# None
            total_scores = fluency * sim * style

        return total_scores.squeeze()

    def scoring(self, news, olds,state_vec):

        # if there is only one word in the input_news, return 0
        if max(len(word.split()) for word in news)==1:
            return 0,-1,-1
        new_embs=self.tokenizer(news,return_tensors='pt',padding=True, truncation=True, max_length=self.max_len)
        new_embs={k:v.to(self.device) for k,v in new_embs.items()}
        old_embs=self.tokenizer(olds,return_tensors='pt',padding=True, truncation=True, max_length=self.max_len)
        old_embs={k:v.to(self.device) for k,v in old_embs.items()}

        fluency_scores = self.fluency_scorer(new_embs)  # input-news, ref_oris-->["I like you"]
        sim_scores = self.semantic_scorer(new_embs, old_embs, state_vec).squeeze()
        style_scores, style_labels = self.style_scorer(news)  # input-news, ref_oris-->["I like you"]

        _scores=self.scorer(fluency_scores,sim_scores,style_scores)

        _score_index=torch.argmax(_scores)
        _score=torch.max(_scores)

        V_score = torch.sigmoid(_score).cpu().detach().numpy()

        # Scale the score to be [-3,3]
        V_score = V_score * 3

        return _score_index,V_score,style_labels