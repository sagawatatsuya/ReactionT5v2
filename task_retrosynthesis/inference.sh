CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top1similartotestall/checkpoint-13618" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top3similartotestall/checkpoint-34639" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top5similartotestall/checkpoint-54934" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top5similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top1similartotestall/checkpoint-13673" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top3similartotestall/checkpoint-34694" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top5similartotestall/checkpoint-54989" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top5similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top1similartotestall/checkpoint-13728" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top3similartotestall/checkpoint-34749" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top5similartotestall/checkpoint-54989" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top5similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top1similartotestall/checkpoint-13871" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top3similartotestall/checkpoint-34892" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top5similartotestall/checkpoint-55187" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top5similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top1similartotestall/checkpoint-14146" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top3similartotestall/checkpoint-35167" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top5similartotestall/checkpoint-55462" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top5similartotestall"