CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling10_seed0" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=10 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling50_seed0" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=50 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling200_seed0" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=200 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling30_seed0" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=30 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling100_seed0" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=100 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0





CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling10_seed1" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=10 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling50_seed1" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=50 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling200_seed1" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=200 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling30_seed1" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=30 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/CompoundT5_finetune_sampling100_seed1" \
    --model_name_or_path='sagawa/CompoundT5' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=100 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1






CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling10_seed0" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=10 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling50_seed0" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=50 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling200_seed0" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=200 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling30_seed0" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=30 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling100_seed0" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=100 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=0





CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling10_seed1" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=10 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling50_seed1" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=50 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling200_seed1" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=200 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling30_seed1" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=30 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --output_dir="finetune/ReactionT5_finetune_sampling100_seed1" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data2/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --sampling_num=100 \
    --save_total_limit=20 \
    --fp16 \
    --input_max_length=200 \
    --seed=1