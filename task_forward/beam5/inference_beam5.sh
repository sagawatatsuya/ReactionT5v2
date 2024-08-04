#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -m a
#$ -m b
#$ -m e
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/12.2 cudnn/8.9 nccl/2.18/2.18.5-1 
source /home/acf15718oa/reactiont5/bin/activate

python prediction.py \
    --input_data="/home/acf15718oa/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --model_name_or_path="/groups/gca50095/sagawa/ReactionT5_neword/forward_reaction_prediction_drop_dup/finetuning_lr2e-5_20epoch_maxinput200/t5/checkpoint-102264" \
    --input_column="input" \
    --input_max_length="150" \
    --output_min_length="2" \
    --output_max_length="181" \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=36 \
    --output_dir="./beam5"