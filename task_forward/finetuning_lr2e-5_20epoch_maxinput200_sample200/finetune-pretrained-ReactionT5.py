import os
import warnings
import sys

import numpy as np
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
sys.path.append("../../")
from utils import seed_everything, get_accuracy_score, preprocess_dataset

# Suppress warnings and disable progress bars
warnings.filterwarnings("ignore")
datasets.utils.logging.disable_progress_bar()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Training script for reaction prediction model."
    )
    parser.add_argument(
        "--train_data_path", type=str, required=True, help="Path to training data CSV."
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        required=True,
        help="Path to validation data CSV.",
    )
    parser.add_argument("--model", type=str, default="t5", help="Model name.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="sagawa/ReactionT5-product-prediction",
        help="The name of a pretrained model or path to a model which you want to finetune on your dataset. You can use your local models or models uploaded to hugging face.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debug mode."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs for training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--input_max_len", type=int, default=400, help="Max input token length."
    )
    parser.add_argument(
        "--target_max_len", type=int, default=150, help="Max target token length."
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="PRODUCT",
        help="Target column name.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        help="Evaluation strategy used during training. Select from 'no', 'steps', or 'epoch'. If you select 'steps', also give --eval_steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Evaluation steps.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="Save strategy used during training. Select from 'no', 'steps', or 'epoch'. If you select 'steps', also give --save_steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save steps.",
    )
    parser.add_argument(
        "--logging_strategy",
        type=str,
        default="epoch",
        help="Logging strategy used during training. Select from 'no', 'steps', or 'epoch'. If you select 'steps', also give --logging_steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Logging steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Limit of saved checkpoints.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable fp16 training.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        default=False,
        help="Disable tqdm.",
    )
    #     parser.add_argument("--multitask", action="store_true", default=False, required=False)
    parser.add_argument(
        "--seed", type=int, default=42, help="Set seed for reproducibility."
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

    train = pd.read_csv(CFG.train_data_path)
    if CFG.sampling_num > 0:
        train = train.sample(n=CFG.sampling_num, random_state=CFG.seed).reset_index(
            drop=True
        )
    valid = pd.read_csv(CFG.valid_data_path)

    for col in ["REACTANT", "REAGENT"]:
        train[col] = train[col].fillna(" ")
        valid[col] = valid[col].fillna(" ")
    train["input"] = "REACTANT:" + train["REACTANT"] + "REAGENT:" + train["REAGENT"]
    valid["input"] = "REACTANT:" + valid["REACTANT"] + "REAGENT:" + valid["REAGENT"]

    if CFG.debug:
        train = train[: int(len(train) / 40)].reset_index(drop=True)
        valid = valid[: int(len(valid) / 40)].reset_index(drop=True)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train[["input", "PRODUCT"]]),
            "validation": Dataset.from_pandas(valid[["input", "PRODUCT"]]),
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

    model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name_or_path).to(device)
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_dataset(examples, CFG),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        CFG.model,
        evaluation_strategy=CFG.evaluation_strategy,
        save_strategy=CFG.save_strategy,
        logging_strategy=CFG.logging_strategy,
        learning_rate=CFG.lr,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size * 4,
        weight_decay=CFG.weight_decay,
        save_total_limit=CFG.save_total_limit,
        num_train_epochs=CFG.epochs,
        predict_with_generate=True,
        fp16=CFG.fp16,
        disable_tqdm=CFG.disable_tqdm,
        push_to_hub=False,
        load_best_model_at_end=True,
    )

    model.config.eval_beams = 5
    model.config.max_length = 150
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

    trainer.train()
    trainer.save_model("./best_model")
