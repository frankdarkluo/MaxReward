from torch.nn import functional as F
import argparse
import numpy as np
from tqdm import tqdm
import math
import string
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

def calc_perplexity_sent_average(model, tokenizer, predictions, device):
    """
        Calculates the sentence-level perplexity.
    """
    with torch.no_grad():
        ppls = []
        error_s = 0
        pbar = tqdm(predictions, desc='Calculating the sentence-level perplexity scores...')
        for i, sample in enumerate(pbar):
            tokens = sample.strip()
            tokens = tokens.split()
            sample = "".join(
                [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
            tokens_tensor = tokenizer.encode(sample, return_tensors="pt", truncation=True, max_length=512).to(device)
            if tokens_tensor.shape[1] > 1:
                loss = model(tokens_tensor, labels=tokens_tensor)[0]
                ppl = np.exp(loss.cpu().detach().numpy())
                if math.isnan(ppl):
                    error_s += 1
                else:
                    ppls.append(ppl)
    return sum(ppls)/len(ppls), error_s


def calc_perplexity_token_average(model, tokenizer, predictions, device):
    """
        Calculates the token-level perplexity.
    """
    with torch.no_grad():
        nlls = []
        total = 0
        error_s = 0
        pbar = tqdm(predictions, desc='Calculating the token-level perplexity scores...')
        for i, sample in enumerate(pbar):
            tokens = sample.strip()
            tokens = tokens.split()
            sample = "".join(
                [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
            text_ids = tokenizer.encode(sample, return_tensors="pt", truncation=True, max_length=512).to(device)
            if text_ids.shape[1] > 1:
                input_ids = text_ids[:, :-1]
                target_ids = text_ids[:, 1:]
                outputs = model(input_ids)
                preds = outputs.logits[0]
                calc_loss = F.nll_loss(F.log_softmax(preds, dim=1), target_ids[0])
                neg_log_likelihood = calc_loss * input_ids.shape[1]
                total += input_ids.shape[1]
                nlls.append(neg_log_likelihood)
            else:
                error_s += 1
        ppl = torch.exp(torch.stack(nlls).sum() / total)
        return ppl.item(), error_s


def main():
    # Parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--gen_path', type=str, default='../output.txt')
    parser.add_argument('--ppl_model', default='gpt2-large')

    args = parser.parse_args()
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'*** Loading the PPL model ({args.ppl_model})...')
    ppl_model = AutoModelForCausalLM.from_pretrained(args.ppl_model).eval().to(device)
    ppl_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model)

    tmp = args.dataset

    predictions=open(args.gen_path,'r',encoding='utf8').readlines()

    # Calculuate the sentence-level and token-level perplexity scores of predictions
    avg_ppl_tok, err_tok_level = calc_perplexity_token_average(ppl_model, ppl_tokenizer, predictions, device)
    # avg_ppl_sent, err_sent_level = calc_perplexity_sent_average(ppl_model, ppl_tokenizer, predictions, device)


    print(f'{tmp}... token-ppl: {avg_ppl_tok}')

if __name__ == "__main__":
    main()

