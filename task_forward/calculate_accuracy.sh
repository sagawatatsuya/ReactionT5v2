python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling10/checkpoint-65/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling10/checkpoint-65" \
    --num_beams=5 \
    --target_col="PRODUCT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling30/checkpoint-105/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling30/checkpoint-105" \
    --num_beams=5 \
    --target_col="PRODUCT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling50/checkpoint-100/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling50/checkpoint-100" \
    --num_beams=5 \
    --target_col="PRODUCT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling100/checkpoint-200/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling100/checkpoint-200" \
    --num_beams=5 \
    --target_col="PRODUCT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling200/checkpoint-400/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_forward/finetune/CompoundT5_finetune_sampling200/checkpoint-400" \
    --num_beams=5 \
    --target_col="PRODUCT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_MIT/MIT_separated/test.csv"