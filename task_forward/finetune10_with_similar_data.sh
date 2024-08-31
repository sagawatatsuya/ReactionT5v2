python finetune.py \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=20 \
    --batch_size=2 \
    --train_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/train.csv' \
    --valid_data_path='/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/val.csv' \
    --similar_reaction_data_path="/data1/ReactionT5_neword/task_forward/output_uspto_test/similar_reactions.csv" \
    --sampling_num=10 \
    --fp16 \
    --input_max_length=200