CUDA_VISIBLE_DEVICES=2,3 python prediction.py \
    --input_data_YIELD="/data1/ReactionT5_neword/data/C_N_yield/MFF_Test1/test.csv" \
    --input_max_length="150" \
    --model_name_or_path="/data1/ReactionT5_neword/task_yield_generation/400epoch/checkpoint-19008" \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=12 \
    --output_dir="./"