#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -m a
#$ -m b
#$ -m e
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/12.2 cudnn/8.9 nccl/2.18/2.18.5-1 
source /home/acf15718oa/reactiont5/bin/activate

python prediction.py \
    --input_data_FORWARD="/home/acf15718oa/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv" \
    --input_data_RETROSYNTHESIS="/home/acf15718oa/ReactionT5_neword/data/USPTO_50k/test.csv" \
    --input_data_YIELD="/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv" \
    --input_max_length="150" \
    --model_name_or_path="/home/acf15718oa/ReactionT5_neword/multitask_no_pretraining/t5/checkpoint-565160" \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="./"