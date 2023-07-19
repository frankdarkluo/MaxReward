#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python3 ddp_train.py \
--style_weight 12 \
--sem_weight 3 \
--fluency_weight 2 \
--bsz 4 \
--buffer_size 8 \
--direction 0-1  \
--plm_name ../EleutherAI/gpt-neo-125M \
--topk 5 \
--max_steps 2 \
--output_dir yelp/ \
--dst yelp \
--max_len 16 \
--seed 42 \
--setting zero-shot