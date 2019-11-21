CUDA_VISIBLE_DEVICES=1 python main.py \
    --output_dir ./results/ \
    --log_dir ./results/ \
    --content_dir ./contents/ \
    --mode inference \
    --pre_trained True \
    --checkpoint ./starry-night_in/model-40000 \
    --normalizer in