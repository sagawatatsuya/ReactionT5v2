python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top1similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top1similartotestall/checkpoint-13618" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top3similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top3similartotestall/checkpoint-34639" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top5similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling10_top5similartotestall/checkpoint-54934" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top1similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top1similartotestall/checkpoint-13673" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top3similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top3similartotestall/checkpoint-34694" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top5similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling30_top5similartotestall/checkpoint-54989" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top1similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top1similartotestall/checkpoint-13728" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top3similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top3similartotestall/checkpoint-34749" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top5similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling50_top5similartotestall/checkpoint-54989" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top1similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top1similartotestall/checkpoint-13871" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top3similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top3similartotestall/checkpoint-34892" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top5similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling100_top5similartotestall/checkpoint-55187" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top1similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top1similartotestall/checkpoint-14146" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top3similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top3similartotestall/checkpoint-35167" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"

python calculate_accuracy.py \
    --input_data="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top5similartotestall/output.csv" \
    --model_name_or_path="/data1/ReactionT5_neword/task_retrosynthesis/finetune/t5_finetune_sampling200_top5similartotestall/checkpoint-55462" \
    --num_beams=5 \
    --target_col="REACTANT" \
    --target_data="/data1/ReactionT5_neword/data/USPTO_50k/test.csv"
