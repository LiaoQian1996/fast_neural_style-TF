#!/usr/bin/env bash
python main.py \
    --output_dir starry-night_bn_submean/ \
    --content_dir home/liaoqian/DATA/COC/O_train2014_1000/ \
    --style_dir ./styles/starry-night.jpg \
    --mode train \
    --pre_trained False \
    --normalizer bn \  # bn or in
    --batch_size 4 \
    --crop_size 256 \
    --lr 0.001 \
    --max_iter 40000 \
    --decay_step 20000 \
    --decay_rate 0.1 \
    --save_freq 1000 \
    --display_freq 50 \
    --summary_freq 100 
