CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/t5_finetune_sampling10_similarall/checkpoint-90913" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/t5_finetune_sampling10_similarall/"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/t5_finetune_sampling10_similartotestall/checkpoint-9824" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/t5_finetune_sampling10_similartotestall"

