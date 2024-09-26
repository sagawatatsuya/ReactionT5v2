CUDA_VISIBLE_DEVICES=0 python search_similar_reaction.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/output_ord/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_retrosynthesis/output_ord/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test/embedding_mean.npy" \
    --batch_size=64 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test" \
    --top_k=1 \
    --similarity_threshold=0.8

CUDA_VISIBLE_DEVICES=0 python search_similar_reaction.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/output_ord/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_retrosynthesis/output_ord/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test/embedding_mean.npy" \
    --batch_size=64 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test" \
    --top_k=3 \
    --similarity_threshold=0.8

CUDA_VISIBLE_DEVICES=0 python search_similar_reaction.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/output_ord/input_data.csv" \
    --target_embedding="/data1/ReactionT5_neword/task_retrosynthesis/output_ord/embedding_mean.npy" \
    --query_embedding="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test/embedding_mean.npy" \
    --batch_size=64 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test" \
    --top_k=5 \
    --similarity_threshold=0.8