CUDA_VISIBLE_DEVICES=0 python get_distance.py \
    --input_data="/data1/ReactionT5_neword/task_yield/output_test1_test/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_yield/output_ord_test1/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_yield/output_test1_test/embedding_mean.npy" \
    --batch_size=64 \
    --output_dir="/data1/ReactionT5_neword/task_yield/output_test1_test/"

CUDA_VISIBLE_DEVICES=0 python get_distance.py \
    --input_data="/data1/ReactionT5_neword/task_yield/output_test2_test/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_yield/output_ord_test2/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_yield/output_test2_test/embedding_mean.npy" \
    --batch_size=64 \
    --output_dir="/data1/ReactionT5_neword/task_yield/output_test2_test/"

CUDA_VISIBLE_DEVICES=0 python get_distance.py \
    --input_data="/data1/ReactionT5_neword/task_yield/output_test3_test/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_yield/output_ord_test3/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_yield/output_test3_test/embedding_mean.npy" \
    --batch_size=64 \
    --output_dir="/data1/ReactionT5_neword/task_yield/output_test3_test/"

CUDA_VISIBLE_DEVICES=0 python get_distance.py \
    --input_data="/data1/ReactionT5_neword/task_yield/output_test4_test/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_yield/output_ord_test4/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_yield/output_test4_test/embedding_mean.npy" \
    --batch_size=64 \
    --output_dir="/data1/ReactionT5_neword/task_yield/output_test4_test/"