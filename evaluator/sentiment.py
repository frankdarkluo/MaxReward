from transformers import pipeline,RobertaTokenizer,RobertaForSequenceClassification
import torch
import string
import sys
sys.path.append("../")
import argparse
from model_utils.functions import softmax
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name1='distilbert-base-uncased-finetuned-sst-2-english'
model_name2="siebert/sentiment-roberta-large-english"
pipeline_classifier = pipeline("sentiment-analysis",model=model_name1)
sty_tokenizer = RobertaTokenizer.from_pretrained(model_name2)
sty_model = RobertaForSequenceClassification.from_pretrained(model_name2).to(device)
#classifier = pipeline(model="EleutherAI/gpt-neo-1.3B")

pos=0
neg=0

parser = argparse.ArgumentParser()
parser.add_argument('--gen_path', default='../output.txt', type=str)
args=parser.parse_args()

def classifier(text):
    inputs = sty_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = sty_model(**inputs).logits
    softmax_logits = softmax(logits)
    outputs = {}
    predicted_class_id = softmax_logits.argmax().item()
    outputs['label'] = sty_model.config.id2label[predicted_class_id]
    outputs['score'] = softmax_logits.squeeze()[predicted_class_id]

    return [outputs]


with open(args.gen_path,'r',encoding='utf8') as f:
    datas=f.readlines()

    for idx,data in enumerate(datas):
        tokens = data.strip()
        if idx< 500:
            # tokens=tokens.split()
            # tokens = "".join(
            #     [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
            res=classifier(tokens)
            if res[0]['label'].lower()=='positive':
                pos+=1

            else:neg+=1
        else:
            res = classifier(tokens)
            if res[0]['label'].lower() == 'negative':
                pos += 1
            else:
                neg += 1
    #
    length=idx+1 # 500

print("POS ACC is {}".format(pos/length))

