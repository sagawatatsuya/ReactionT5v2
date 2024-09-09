CUDA_VISIBLE_DEVICES=3 python generate_embedding.py \
    --input_data="/data1/ReactionT5_neword/data/C_N_yield/MFF_Test1/train.csv" \
    --model_name_or_path="sagawa/ReactionT5v2-forward" \
    --input_max_length=150 \
    --batch_size=64 \
    --output_dir="output_test1"