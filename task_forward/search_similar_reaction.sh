CUDA_VISIBLE_DEVICES=0 python search_similar_reaction.py \
    --input_data="/data1/ReactionT5_neword/task_forward/output_ord/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_forward/output_ord/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_forward/output_uspto_test/embedding_mean.npy" \
    --batch_size=256 \
    --output_dir="/data1/ReactionT5_neword/task_forward/output_uspto_test"