CUDA_VISIBLE_DEVICES=2 python reaction_sampling.py \
    --output_dir="finetune/CompoundT5_finetune_sampling30" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=10 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200

CUDA_VISIBLE_DEVICES=2 python reaction_sampling.py \
    --output_dir="finetune/CompoundT5_finetune_sampling30" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=30 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200

CUDA_VISIBLE_DEVICES=2 python reaction_sampling.py \
    --output_dir="finetune/CompoundT5_finetune_sampling30" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=50 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200

CUDA_VISIBLE_DEVICES=2 python reaction_sampling.py \
    --output_dir="finetune/CompoundT5_finetune_sampling30" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=100 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200

CUDA_VISIBLE_DEVICES=2 python reaction_sampling.py \
    --output_dir="finetune/CompoundT5_finetune_sampling30" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=200 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200