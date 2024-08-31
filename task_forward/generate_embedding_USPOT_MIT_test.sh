CUDA_VISIBLE_DEVICES=1 python generate_embedding.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="sagawa/ReactionT5v2-forward" \
    --input_max_length=150 \
    --batch_size=16 \
    --output_dir="output_uspto_test"