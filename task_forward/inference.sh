CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling10_top1similartotestall/checkpoint-208307" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling10_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling10_top3similartotestall/checkpoint-455532" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling10_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling10_top5similartotestall/checkpoint-662079" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling10_top5similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling30_top1similartotestall/checkpoint-246311" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling30_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling30_top3similartotestall/checkpoint-455642" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling30_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling30_top5similartotestall/checkpoint-662189" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling30_top5similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling50_top1similartotestall/checkpoint-208527" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling50_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling50_top3similartotestall/checkpoint-455752" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling50_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling50_top5similartotestall/checkpoint-662299" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling50_top5similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling100_top1similartotestall/checkpoint-208802" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling100_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling100_top3similartotestall/checkpoint-456027" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling100_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling100_top5similartotestall/checkpoint-662574" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling100_top5similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top1similartotestall/checkpoint-228384" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top1similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top3similartotestall/checkpoint-456577" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top3similartotestall"

CUDA_VISIBLE_DEVICES=1 python prediction.py \
    --input_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top5similartotestall/checkpoint-663124" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=24 \
    --output_dir="/data1/ReactionT5_neword/task_forward/finetune/t5_finetune_sampling200_top5similartotestall"