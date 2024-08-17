#!/bin/bash

#$ -l rt_AF=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -m a
#$ -m b
#$ -m e
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/12.2 cudnn/8.9 nccl/2.18/2.18.3-1 hpcx/2.12
source /home/acf15718oa/third/bin/activate
export TRANSFORMERS_CACHE='/groups/gca50095/sagawa/cache/'
export HF_DATASETS_CACHE="/groups/gca50095/sagawa/cache/"

wd=/home/acf15718oa/train_with_deepspeed/hp_search

cd $wd


export OMP_NUM_THREADS=2
export NGPU_PER_NODE=4


# launch on master node
node_rank=0
torchrun --nproc_per_node $NGPU_PER_NODE --nnodes $NHOSTS --node_rank $node_rank --master_addr `hostname` train_with_deepspeed_check.py \
    --do_train \
    --do_eval \
    --num_train_epochs="10" \
    --output_dir="./output" \
    --overwrite_output_dir \
    --save_total_limit="2" \
    --deepspeed="/home/acf15718oa/train_with_deepspeed/deepspeed_configs/ds_config_zero0.json" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="32" \
    --learning_rate="0.001" \
    --weight_decay="0.001" \
    --warmup_steps="10000" \
    --logging_steps="32" \
    --save_steps="4096" \
    --eval_steps="4096" \
    --config_name="/home/acf15718oa/T5configs/small" \
    --tokenizer_name="/home/acf15718oa/T5configs/small" \
    --train_files_dir="/scratch/acf15718oa/preprocessed_ZINC22/" \
    --max_seq_length="512" \
    --num_workers="2"


# finalize
wait
exit 0
