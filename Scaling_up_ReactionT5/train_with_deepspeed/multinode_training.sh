#!/bin/bash

#$ -l rt_AF=2
#$ -l h_rt=6:00:00
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

wd=/home/acf15718oa/train_with_deepspeed

cd $wd


export OMP_NUM_THREADS=2
export NGPU_PER_NODE=4

# launch on slave nodes
node_rank=1
for slave_node in `cat $SGE_JOB_HOSTLIST | awk 'NR != 1 { print }'`; do
echo "######" $slave_node "/" $NHOSTS
qrsh -inherit -V -cwd $slave_node torchrun --nproc_per_node $NGPU_PER_NODE --nnodes $NHOSTS --node_rank $node_rank --master_addr `hostname` train_with_deepspeed_check.py \
    --do_train \
    --do_eval \
    --num_train_epochs="10" \
    --output_dir="/scratch/acf15718oa/train_with_deepspeed/kojima_output" \
    --overwrite_output_dir \
    --save_total_limit="2" \
    --deepspeed="/home/acf15718oa/train_with_deepspeed/deepspeed_configs/ds_config_zero2_offload_optimizer.json" \
    --per_device_train_batch_size="1" \
    --per_device_eval_batch_size="1" \
    --learning_rate="0.001" \
    --weight_decay="0.001" \
    --warmup_steps="0" \
    --logging_steps="1" \
    --save_steps="10" \
    --eval_steps="200" \
    --gradient_checkpointing \
    --config_name="/home/acf15718oa/T5configs/xl" \
    --tokenizer_name="/home/acf15718oa/T5configs/xl" \
    --train_files_dir="/groups/gca50095/sagawa/train_data/" \
    --validation_split_percentage="0.1" \
    --max_seq_length="512" \
    --num_workers="2" &
node_rank=`expr $node_rank + 1`
done

# launch on master node
node_rank=0
torchrun --nproc_per_node $NGPU_PER_NODE --nnodes $NHOSTS --node_rank $node_rank --master_addr `hostname` train_with_deepspeed_check.py \
    --do_train \
    --do_eval \
    --num_train_epochs="10" \
    --output_dir="/scratch/acf15718oa/train_with_deepspeed/kojima_output" \
    --overwrite_output_dir \
    --save_total_limit="2" \
    --deepspeed="/home/acf15718oa/train_with_deepspeed/deepspeed_configs/ds_config_zero2_offload_optimizer.json" \
    --per_device_train_batch_size="1" \
    --per_device_eval_batch_size="1" \
    --learning_rate="0.001" \
    --weight_decay="0.001" \
    --warmup_steps="0" \
    --logging_steps="1" \
    --save_steps="10" \
    --eval_steps="200" \
    --gradient_checkpointing \
    --config_name="/home/acf15718oa/T5configs/xl" \
    --tokenizer_name="/home/acf15718oa/T5configs/xl" \
    --train_files_dir="/groups/gca50095/sagawa/train_data/" \
    --validation_split_percentage="0.1" \
    --max_seq_length="512" \
    --num_workers="2"


# finalize
wait
exit 0
