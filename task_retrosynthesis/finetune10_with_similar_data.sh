CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --output_dir="t5_finetune_sampling10_similarall" \
    --model_name_or_path='sagawa/ReactionT5v2-retrosynthesis' \
    --epochs=20 \
    --batch_size=4 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_50k/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_50k/val.csv' \
    --similar_reaction_data_path="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto/similar_reactions.csv" \
    --sampling_num=10 \
    --fp16 \
    --input_max_length=150

CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --output_dir="t5_finetune_sampling10_similartotestall" \
    --model_name_or_path='sagawa/ReactionT5v2-retrosynthesis' \
    --epochs=20 \
    --batch_size=4 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_50k/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_50k/val.csv' \
    --similar_reaction_data_path="/data1/ReactionT5_neword/task_retrosynthesis/output_uspto_test/similar_reactions.csv" \
    --sampling_num=10 \
    --fp16 \
    --input_max_length=150