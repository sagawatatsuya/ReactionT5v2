CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling10/checkpoint-3" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling10/checkpoint-3"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling10/checkpoint-60" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling10/checkpoint-60"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling30/checkpoint-8" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling30/checkpoint-8"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling30/checkpoint-160" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling30/checkpoint-160"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling50/checkpoint-13" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling50/checkpoint-13"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling50/checkpoint-169" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling50/checkpoint-169"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling100/checkpoint-25" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling100/checkpoint-25"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling100/checkpoint-200" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling100/checkpoint-200"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling200/checkpoint-50" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling200/checkpoint-50"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling200/checkpoint-300" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/CompoundT5_finetune_sampling200/checkpoint-300"