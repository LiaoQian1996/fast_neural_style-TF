#!/usr/bin/env bash
python main.py \
    --output_dir results/ \
    --content_dir DATA/COCO_train2014_1000/ \ # train data image, in .png or .jpg
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
