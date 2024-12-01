CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling10" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=4 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_50k/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_50k/val.csv' \
    --sampling_num=10 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=150


CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling50" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=4 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_50k/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_50k/val.csv' \
    --sampling_num=50 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=150

CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling200" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=4 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_50k/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_50k/val.csv' \
    --sampling_num=200 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=150


CUDA_VISIBLE_DEVICES=3 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling30" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=4 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_50k/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_50k/val.csv' \
    --sampling_num=30 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=150


CUDA_VISIBLE_DEVICES=3 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling100" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=4 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_50k/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_50k/val.csv' \
    --sampling_num=100 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=150
