python main.py \
    --output_dir ./results/ \
    --log_dir ./results/ \
    --content_dir ./contents/ \
    --mode inference \
    --pre_trained True \
    --checkpoint ./starry-night_in/model-40000 \
    --normalizer in \
    --vgg_ckpt ./vgg19/vgg_19.ckpt
