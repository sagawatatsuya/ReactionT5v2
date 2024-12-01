CUDA_VISIBLE_DEVICES=0 python generate_embedding_v5.py \
    --input_data="../data/all_ord_reaction_uniq_with_attr20240506_v3_train.csv" \
    --test_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="sagawa/ReactionT5v2-retrosynthesis" \
    --input_max_length=150 \
    --batch_size=16 \
    --output_dir="output_ord"

CUDA_VISIBLE_DEVICES=0 python generate_embedding_v5.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="sagawa/ReactionT5v2-retrosynthesis" \
    --input_max_length=150 \
    --batch_size=16 \
    --output_dir="output_uspto_test"