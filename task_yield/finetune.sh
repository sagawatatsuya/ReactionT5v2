CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test1/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --download_pretrained_model \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test1_frac10' \
    --patience=20 \
    --sampling_frac=0.1

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test1/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test1_frac5' \
    --patience=20 \
    --sampling_frac=0.05

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test1/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test1_frac1' \
    --patience=20 \
    --sampling_frac=0.01


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test2/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test2_frac10' \
    --patience=20 \
    --sampling_frac=0.1

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test2/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test2_frac5' \
    --patience=20 \
    --sampling_frac=0.05

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test2/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test2_frac1' \
    --patience=20 \
    --sampling_frac=0.01


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test3/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test3_frac10' \
    --patience=20 \
    --sampling_frac=0.1

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test3/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test3_frac5' \
    --patience=20 \
    --sampling_frac=0.05

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test3/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test3_frac1' \
    --patience=20 \
    --sampling_frac=0.01


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test4/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test4_frac10' \
    --patience=20 \
    --sampling_frac=0.1

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test4/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test4_frac5' \
    --patience=20 \
    --sampling_frac=0.05

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test4/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/data1/ReactionT5_neword/task_yield' \
    --epochs=200 \
    --batch_size=16 \
    --input_max_length=300 \
    --output_dir='/data1/ReactionT5_neword/task_yield/finetune/finetune_test4_frac1' \
    --patience=20 \
    --sampling_frac=0.01