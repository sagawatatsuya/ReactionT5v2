#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -m a
#$ -m b
#$ -m e
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/12.2 cudnn/8.9 nccl/2.18/2.18.5-1 
source /home/acf15718oa/reactiont5/bin/activate

python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac20' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac20'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac40' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac40'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac60' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac60'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac80' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac80'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac100' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test1_frac100'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac20' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac20'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac40' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac40'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac60' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac60'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac80' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac80'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac100' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test2_frac100'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac20' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac20'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac40' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac40'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac60' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac60'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac80' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac80'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac100' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test3_frac100'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac20' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac20'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac40' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac40'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac60' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac60'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac80' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac80'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac100' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/finetune/finetune_test4_frac100'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac20' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac20'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac40' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac40'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac60' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac60'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac80' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac80'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac100' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test1_frac100'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac20' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac20'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac40' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac40'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac60' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac60'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac80' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac80'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test2/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac100' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test2_frac100'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac20' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac20'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac40' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac40'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac60' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac60'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac80' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac80'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test3/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac100' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test3_frac100'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac20' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac20'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac40' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac40'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac60' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac60'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac80' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac80'
python prediction.py \
    --data='/home/acf15718oa/ReactionT5_neword/data/C_N_yield/MFF_Test4/test.csv' \
    --model_name_or_path='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac100' \
    --batch_size=32 \
    --output_dir='/groups/gca50095/sagawa/ReactionT5_neword/yield_prediction/train/train_test4_frac100'