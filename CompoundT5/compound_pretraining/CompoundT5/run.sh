python ./new_run_t5_mlm_flax.py --output_dir="./CompoundT5-output" --model_type="t5" --config_name="./CompoundT5-config" --tokenizer_name="./CompoundT5-config" --dataset_name "sagawa/ZINC-canonicalized" --max_seq_length="512" --per_device_train_batch_size="5" --per_device_eval_batch_size="5" --adafactor --learning_rate="0.005" --weight_decay="0.001" --warmup_steps="2000" --overwrite_output_dir --logging_steps="500" --save_steps="100000" --num_train_epochs="30" --do_train --do_eval --eval_steps="100000"