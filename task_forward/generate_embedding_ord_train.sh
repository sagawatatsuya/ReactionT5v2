CUDA_VISIBLE_DEVICES=3 python generate_embedding.py \
    --input_data="../data/all_ord_reaction_uniq_with_attr20240506_v3_train.csv" \
    --model_name_or_path="sagawa/ReactionT5v2-forward" \
    --input_max_length=150 \
    --batch_size=16 \
    --output_dir="output_uspto"