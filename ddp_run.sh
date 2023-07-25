#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 main.py \
--style_weight 12 \
--sem_weight 3 \
--fluency_weight 2 \
--bsz 4 \
--buffer_size 1000 \
--update_interval 100 \
--direction 0-1  \
--plm_name ../EleutherAI/gpt-j-6B \
--topk 5 \
--max_steps 5 \
--output_dir yelp/ \
--dst yelp \
--max_len 16 \
--seed 42 \
--setting zero-shot