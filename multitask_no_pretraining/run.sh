#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -m a
#$ -m b
#$ -m e
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/12.2 cudnn/8.9 nccl/2.18/2.18.5-1 
source /home/acf15718oa/reactiont5/bin/activate

python train.py \
    --model='t5' \
    --epochs=40 \
    --lr=2e-4 \
    --batch_size=32 \
    --input_max_len=300 \
    --target_max_len=150 \
    --weight_decay=0.01 \
    --evaluation_strategy='epoch' \
    --save_strategy='epoch' \
    --logging_strategy='epoch' \
    --save_total_limit=100 \
    --train_data_path_FORWARD='/home/acf15718oa/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path_FORWARD='/home/acf15718oa/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --train_data_path_RETROSYNTHESIS='/home/acf15718oa/ReactionT5_neword/data/USPTO_50k/train.csv' \
    --valid_data_path_RETROSYNTHESIS='/home/acf15718oa/ReactionT5_neword/data/USPTO_50k/val.csv' \
    --train_data_path_YIELD='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/train.csv' \
    --valid_data_path_YIELD='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --disable_tqdm \
    --pretrained_model_name_or_path='sagawa/ZINC-t5'
