import torch
import random
from transformers import RobertaTokenizer, RobertaForMaskedLM

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForMaskedLM.from_pretrained("roberta-large")

def nucleus_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')

    return logits

def get_edit_probabilities(sentence):
    # Compute a proxy for edit probabilities by masking each position
    edit_probs = []
    for i in range(len(sentence.split())):
        masked_sentence = " ".join([word if idx != i else tokenizer.mask_token for idx, word in enumerate(sentence.split())])
        inputs = tokenizer(masked_sentence, return_tensors="pt")
        mask_position = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()

        with torch.no_grad():
            logits = model(**inputs).logits[0, mask_position]

        edit_prob = torch.softmax(logits, dim=-1).max().item()
        edit_probs.append(edit_prob)

    return torch.tensor(edit_probs)

def edit_sentence(sentence, p=0.9, k=3):
    # Get edit probabilities
    edit_probs = get_edit_probabilities(sentence)

    # Apply nucleus sampling to edit probabilities
    sampled_edit_probs = nucleus_sampling(edit_probs, p)

    # Select top-k positions with the highest probabilities
    top_k_positions = torch.topk(sampled_edit_probs, k).indices

    # Edit tokens at the selected positions
    edited_sentence = []
    for position, word in enumerate(sentence.split()):
        if position in top_k_positions:
            masked_sentence = " ".join([word if idx != position else tokenizer.mask_token for idx, _ in enumerate(sentence.split())])
            inputs = tokenizer(masked_sentence, return_tensors="pt")
            mask_position = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                logits = model(**inputs).logits[0, mask_position]

            new_token_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            new_word = tokenizer.decode(new_token_id)
        else:
            new_word = word

        edited_sentence.append(new_word)

    return " ".join(edited_sentence)

sentence = "This is a test sentence."
edited_sentence = edit_sentence(sentence)
print(edited_sentence)
