CUDA_VISIBLE_DEVICES=0 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/t5_finetune_sampling10_similarall/checkpoint-45457" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=16 \
    --output_dir="/data1/ReactionT5_neword/task_forward/t5_finetune_sampling10_similarall/"

CUDA_VISIBLE_DEVICES=0 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/t5_finetune_sampling10_similartotestall/checkpoint-4777" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=16 \
    --output_dir="/data1/ReactionT5_neword/task_forward/t5_finetune_sampling10_similartotestall/"