from transformers import pipeline,GPT2Tokenizer,AutoModelForCausalLM,AutoTokenizer
import argparse
import torch
from utils.functions import softmax

prefix = """
    Sentence: i do not intend to be mean
    Formality: {formal}

    #####

    Sentence: ohhh i don't intend to be mean ..
    Formality: {informal}

    #####

    Sentence: what 're u doing here ?
    Formality: {informal}

    #####

    Sentence: people are having coffee , lunch, playing in the park , playing and talking .
    Formality: {formal}

    #####

    Sentence: well, that is simply the manner it is done, i suppose.
    Formality: {formal}

    ####

    Sentence: well that is just the way it is I guess.
    Formality: {informal}

    ####

    Sentence: hello, i am in NYC and i could assist you if you need.
    Formality: {formal}

    """
postfix = "Formality: {"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate(model,tokenizer,args):
    if args.direction == '0-1': postfix = '0'
    else: postfix = '1'

    output_path='../output/'+args.task+'_'+args.direction+'_'+args.model_name.split('/')[-1]+'.txt'
    data_path='../data/gyafc_500/test.'+postfix
    with open(data_path,'r',encoding='utf8') as f, open(output_path,'no_fluency',encoding='utf8') as of:
        datas=f.readlines()
        for line in datas:
            line=line.strip()

            input="Sentence: "+str(line)+"\n"
            postfix="Sentiment: {"
            if args.task == 'few-shot':
                prompt = prefix+ input +postfix
            else:
                prompt = input+postfix

            indexed_tokens = tokenizer.encode(prompt)
            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens]).cuda()
            # Set the model in evaluation mode to deactivate the DropOut modules
            model.eval()

            # Predict all tokens
            with torch.no_grad():
                outputs = model(tokens_tensor)
                predictions = outputs[0]

            probs = predictions[0, -1, :]

            pos_logits = probs[tokenizer.encode('formal')]
            neg_logits = probs[tokenizer.encode('informal')]

            emo_logits = torch.concat([neg_logits, pos_logits])
            softmax_emo_logits = softmax(emo_logits)

            pos_prob = softmax_emo_logits[1]
            neg_prob = softmax_emo_logits[0]

            if torch.argmax(softmax_emo_logits) == 0 and neg_prob >= 0.6:
                label = 'informal'
            elif torch.argmax(softmax_emo_logits) == 1 and pos_prob >= 0.6:
                label = 'formal'
            else:
                label = 'neutral'
            # if args.task == 'few-shot':
            #     # output_sent=''.join(output_sent.split('\n')[X+1:X+2])
            #     output_sent = ''.join(output_sent.split('#####')[-1].split('\n\n')[-1])
            # else:
            #     output_sent = ''.join(output_sent.split(postfix)[-1])
            of.write('Generated:' + '\n')
            of.write(input+postfix+label+'}'+'\n\n')
            of.flush()

def main(args):
    model_name=args.model_name
    model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generate(model, tokenizer, args)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='EleutherAI/gpt-neo-1.3B', help="super large PLM")
    parser.add_argument("--task",type=str, default='few-shot')
    parser.add_argument("--direction",type=str, default='0-1')
    args=parser.parse_args()
    print("let's start")
    main(args)
