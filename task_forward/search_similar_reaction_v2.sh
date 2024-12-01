CUDA_VISIBLE_DEVICES=0 python search_similar_reaction_v2.py \
    --input_data="/data1/ReactionT5_neword/task_forward/output_ord/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_forward/output_ord/embedding_mean_v2.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_forward/output_uspto_test/embedding_mean_v2.npy" \
    --batch_size=256 \
    --output_dir="/data1/ReactionT5_neword/task_forward/output_uspto_test" \
    --top_k=1

CUDA_VISIBLE_DEVICES=0 python search_similar_reaction_v2.py \
    --input_data="/data1/ReactionT5_neword/task_forward/output_ord/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_forward/output_ord/embedding_mean_v2.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_forward/output_uspto_test/embedding_mean_v2.npy" \
    --batch_size=256 \
    --output_dir="/data1/ReactionT5_neword/task_forward/output_uspto_test" \
    --top_k=3

CUDA_VISIBLE_DEVICES=0 python search_similar_reaction_v2.py \
    --input_data="/data1/ReactionT5_neword/task_forward/output_ord/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_forward/output_ord/embedding_mean_v2.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_forward/output_uspto_test/embedding_mean_v2.npy" \
    --batch_size=256 \
    --output_dir="/data1/ReactionT5_neword/task_forward/output_uspto_test" \
    --top_k=5