#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 SAHC.py \
--style_weight 12 \
--sem_weight 3 \
--fluency_weight 2 \
--direction 0-1  \
--style_mode plm \
--class_name ../EleutherAI/gpt-j-6B \
--topk 50 \
--max_steps 5 \
--output_dir yelp/ \
--dst yelp \
--semantic_mode kw-sent \
--max_len 16 \
--seed 42 \
--setting zero-shot \
--ablation None \
--early_stop True

