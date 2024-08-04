#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -m a
#$ -m b
#$ -m e
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/12.2 cudnn/8.9 nccl/2.18/2.18.5-1 
source /home/acf15718oa/reactiont5/bin/activate

python finetune-pretrained-ReactionT5.py \
    --model='t5' \
    --epochs=50 \
    --lr=2e-5 \
    --batch_size=4 \
    --input_max_len=200 \
    --target_max_len=150 \
    --evaluation_strategy='epoch' \
    --save_strategy='epoch' \
    --logging_strategy='epoch' \
    --save_total_limit=2 \
    --train_data_path='/home/acf15718oa/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/home/acf15718oa/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --disable_tqdm \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/forward_reaction_prediction_drop_dup/t5/checkpoint-486000' \
    --sampling_num=200
