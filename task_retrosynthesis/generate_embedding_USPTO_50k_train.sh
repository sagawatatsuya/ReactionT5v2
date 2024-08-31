CUDA_VISIBLE_DEVICES=3 python generate_embedding.py \
    --input_data="../data/USPTO_MIT/MIT_separated/train.csv" \
    --model_name_or_path="sagawa/ReactionT5v2-retrosynthesis" \
    --input_max_length=150 \
    --batch_size=16 \
    --output_dir="output_uspto"