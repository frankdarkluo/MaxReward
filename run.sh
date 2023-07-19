#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python3 inference.py \
--style_weight 8 \
--sem_weight 3 \
--fluency_weight 2 \
--bsz 1 \
--direction 1-0  \
--plm_name ../EleutherAI/gpt-j-6B \
--topk 30 \
--max_steps 5 \
--output_dir amazon \
--dst amazon \
--max_len 16 \
--seed 42 \
--setting zero-shot

CUDA_VISIBLE_DEVICES=3 python3 inference.py \
--style_weight 8 \
--sem_weight 3 \
--fluency_weight 2 \
--bsz 1 \
--direction 0-1  \
--plm_name ../EleutherAI/gpt-j-6B \
--topk 30 \
--max_steps 5 \
--output_dir amazon \
--dst amazon \
--max_len 16 \
--seed 42 \
--setting zero-shot