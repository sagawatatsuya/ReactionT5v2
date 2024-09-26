CUDA_VISIBLE_DEVICES=0 python get_distance.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_retrosynthesis/output_ord/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test/embedding_mean.npy" \
    --batch_size=64 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test"