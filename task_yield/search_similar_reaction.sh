CUDA_VISIBLE_DEVICES=0 python search_similar_reaction.py \
    --input_data="/data1/ReactionT5_neword/task_yield/output_ord/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_yield/output_ord/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_yield/output_test1/embedding_mean.npy" \
    --batch_size=64 \
    --output_dir="./"