
CUDA_VISIBLE_DEVICES=0 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top1similartotestall/checkpoint-228384" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top1similartotestall"

CUDA_VISIBLE_DEVICES=0 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top3similartotestall/checkpoint-456577" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top3similartotestall"

CUDA_VISIBLE_DEVICES=0 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top5similartotestall/checkpoint-663124" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top5similartotestall"