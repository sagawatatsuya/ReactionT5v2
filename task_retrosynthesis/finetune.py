import os
import warnings
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
import datasets
from datasets import Dataset, DatasetDict
import argparse

sys.path.append("../")
from utils import seed_everything, get_accuracy_score, preprocess_dataset, filter_out
from train import preprocess_df

# Suppress warnings and disable progress bars
warnings.filterwarnings("ignore")
datasets.utils.logging.disable_progress_bar()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="The path to data used for training. CSV file that contains ['REACTANT', 'PRODUCT'] columns is expected.",
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        required=True,
        help="The path to data used for validation. CSV file that contains ['REACTANT', 'PRODUCT'] columns is expected.",
    )
    parser.add_argument(
        "--similar_reaction_data_path",
        type=str,
        required=False,
        help="Path to similar data CSV.",
    )
    parser.add_argument("--output_dir", type=str, default="t5", help="Path of the output directory.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        default="sagawa/ReactionT5v2-retrosynthesis",
        help="The name of a pretrained model or path to a model which you want to finetune on your dataset. You can use your local models or models uploaded to hugging face.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        required=False,
        help="Use debug mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        required=False,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--input_max_length",
        type=int,
        default=150,
        required=False,
        help="Max input token length.",
    )
    parser.add_argument(
        "--target_max_length",
        type=int,
        default=150,
        required=False,
        help="Max target token length.",
    )
    parser.add_argument(
        "--eval_beams",
        type=int,
        default=5,
        help="Number of beams used for beam search during evaluation.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="REACTANT",
        help="Target column name.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        required=False,
        help="weight_decay used for trainer",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        required=False,
        help="Evaluation strategy used during training. Select from 'no', 'steps', or 'epoch'. If you select 'steps', also give --eval_steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        required=False,
        help="Number of update steps between two evaluations",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        required=False,
        help="Save strategy used during training. Select from 'no', 'steps', or 'epoch'. If you select 'steps', also give --save_steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        required=False,
        default=500,
        help="Number of steps between two saving",
    )
    parser.add_argument(
        "--logging_strategy",
        type=str,
        default="epoch",
        required=False,
        help="Logging strategy used during training. Select from 'no', 'steps', or 'epoch'. If you select 'steps', also give --logging_steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        required=False,
        default=500,
        help="Number of steps between two logging",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        required=False,
        help="Limit of the number of saved checkpoints. If limit is reached, the oldest checkpoint will be deleted.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        required=False,
        help="Use fp16 during training",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        default=False,
        required=False,
        help="Disable tqdm during training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Set seed for reproducibility.",
    )
    parser.add_argument(
        "--sampling_num",
        type=int,
        default=-1,
        help="Number of samples used for training. If you want to use all samples, set -1.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    CFG = parse_args()
    CFG.disable_tqdm = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=CFG.seed)

    train = preprocess_df(filter_out(pd.read_csv(CFG.train_data_path), ["REACTANT", "PRODUCT"]))
    valid = preprocess_df(filter_out(pd.read_csv(CFG.valid_data_path), ["REACTANT", "PRODUCT"]))

    if CFG.sampling_num > 0:
        train = train.sample(n=CFG.sampling_num, random_state=CFG.seed).reset_index(
            drop=True
        )

    if CFG.similar_reaction_data_path:
        similar = preprocess_df(filter_out(
            pd.read_csv(CFG.similar_reaction_data_path), ["REACTANT", "PRODUCT"])
        )
        print(len(train))
        train = pd.concat([train, similar], ignore_index=True)
        print(len(train))

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train[["input", "REACTANT"]]),
            "validation": Dataset.from_pandas(valid[["input", "REACTANT"]]),
        }
    )

    # load tokenizer
    try:  # load pretrained tokenizer from local directory
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.abspath(CFG.model_name_or_path), return_tensors="pt"
        )
    except:  # load pretrained tokenizer from huggingface model hub
        tokenizer = AutoTokenizer.from_pretrained(
            CFG.model_name_or_path, return_tensors="pt"
        )
    CFG.tokenizer = tokenizer

    # load model
    try:  # load pretrained model from local directory
        model = AutoModelForSeq2SeqLM.from_pretrained(
            os.path.abspath(CFG.model_name_or_path)
        )
    except:  # load pretrained model from huggingface model hub
        model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name_or_path)
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_dataset(examples, CFG),
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        CFG.output_dir,
        evaluation_strategy=CFG.evaluation_strategy,
        eval_steps=CFG.eval_steps,
        save_strategy=CFG.save_strategy,
        save_steps=CFG.save_steps,
        logging_strategy=CFG.logging_strategy,
        logging_steps=CFG.logging_steps,
        learning_rate=CFG.lr,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        weight_decay=CFG.weight_decay,
        save_total_limit=CFG.save_total_limit,
        num_train_epochs=CFG.epochs,
        predict_with_generate=True,
        fp16=CFG.fp16,
        disable_tqdm=CFG.disable_tqdm,
        push_to_hub=False,
        load_best_model_at_end=True,
    )

    model.config.eval_beams = CFG.eval_beams
    model.config.max_length = CFG.target_max_length
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: get_accuracy_score(eval_preds, CFG),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model("./best_model")
