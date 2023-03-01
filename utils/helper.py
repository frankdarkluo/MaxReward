# -*- coding: utf-8 -*-

import torch

from torch import cuda
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from model_utils.dataset import collate_fn

def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


device = 'cuda' if cuda.is_available() else 'cpu'


def optimize(opt, loss, retain_graph=False):
    opt.zero_grad()
    loss.backward(retain_graph=retain_graph)
    opt.step()


