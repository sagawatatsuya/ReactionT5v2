CUDA_VISIBLE_DEVICES=2 python finetune.py \
    --output_dir="finetune/t5_finetune_sampling10_top1similartotestall" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --similar_reaction_data_path="/data1/ReactionT5_neword/task_forward/output_uspto_test/top1_similar_reactions.csv" \
    --sampling_num=10 \
    --fp16 \
    --input_max_length=200

CUDA_VISIBLE_DEVICES=2 python finetune.py \
    --output_dir="finetune/t5_finetune_sampling10_top3similartotestall" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --similar_reaction_data_path="/data1/ReactionT5_neword/task_forward/output_uspto_test/top3_similar_reactions.csv" \
    --sampling_num=10 \
    --fp16 \
    --input_max_length=200

CUDA_VISIBLE_DEVICES=2 python finetune.py \
    --output_dir="finetune/t5_finetune_sampling10_top5similartotestall" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --similar_reaction_data_path="/data1/ReactionT5_neword/task_forward/output_uspto_test/top5_similar_reactions.csv" \
    --sampling_num=10 \
    --fp16 \
    --input_max_length=200


CUDA_VISIBLE_DEVICES=2 python finetune.py \
    --output_dir="finetune/t5_finetune_sampling30_top1similartotestall" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --similar_reaction_data_path="/data1/ReactionT5_neword/task_forward/output_uspto_test/top1_similar_reactions.csv" \
    --sampling_num=30 \
    --fp16 \
    --input_max_length=200

CUDA_VISIBLE_DEVICES=2 python finetune.py \
    --output_dir="finetune/t5_finetune_sampling30_top3similartotestall" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --similar_reaction_data_path="/data1/ReactionT5_neword/task_forward/output_uspto_test/top3_similar_reactions.csv" \
    --sampling_num=30 \
    --fp16 \
    --input_max_length=200

CUDA_VISIBLE_DEVICES=2 python finetune.py \
    --output_dir="finetune/t5_finetune_sampling30_top5similartotestall" \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --similar_reaction_data_path="/data1/ReactionT5_neword/task_forward/output_uspto_test/top5_similar_reactions.csv" \
    --sampling_num=30 \
    --fp16 \
    --input_max_length=200