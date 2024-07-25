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
from utils import seed_everything, canonicalize, space_clean, get_accuracy_score, preprocess_dataset

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
    parser.add_argument("--test_data_path", type=str, help="Path to test data CSV.")
    parser.add_argument("--model", type=str, default="t5", help="Model name.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Pretrained model path or name.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debug mode."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--input_max_len",
        type=int,
        default=400,
        help="Max input token length.",
    )
    parser.add_argument(
        "--target_max_len",
        type=int,
        default=150,
        help="Max target token length.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def preprocess_df(df):
    """Preprocess the dataframe by filling NaNs, dropping duplicates, and formatting the input."""
    df = df[~df["PRODUCT"].isna()]
    for col in ["CATALYST", "REACTANT", "REAGENT", "SOLVENT", "PRODUCT"]:
        df[col] = df[col].fillna(" ")
    df = df[df["REACTANT"] != " "]
    df = (
        df[["REACTANT", "PRODUCT", "CATALYST", "REAGENT", "SOLVENT"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df["REAGENT"] = df["CATALYST"] + "." + df["REAGENT"] + "." + df["SOLVENT"]
    df["REAGENT"] = df["REAGENT"].apply(lambda x: space_clean(x))
    df["REAGENT"] = df["REAGENT"].apply(lambda x: canonicalize(x) if x != " " else " ")
    df["input"] = "REACTANT:" + df["REACTANT"] + "REAGENT:" + df["REAGENT"]
    return df


if __name__ == "__main__":
    CFG = parse_args()
    CFG.disable_tqdm = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=CFG.seed)

    # Load and preprocess data
    train = preprocess_df(pd.read_csv(CFG.train_data_path))
    valid = preprocess_df(pd.read_csv(CFG.valid_data_path))
    train["pair"] = train["input"] + " - " + train["PRODUCT"]
    valid["pair"] = valid["input"] + " - " + valid["PRODUCT"]
    valid = valid[~valid["pair"].isin(train["pair"])].reset_index(drop=True)
    train.to_csv("train.csv", index=False)
    valid.to_csv("valid.csv", index=False)

    if CFG.test_data_path:
        test = preprocess_df(pd.read_csv(CFG.test_data_path))
        test["pair"] = test["input"] + " - " + test["PRODUCT"]
        test = test[~test["pair"].isin(train["pair"])].reset_index(drop=True)
        test.to_csv("test.csv", index=False)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train[["input", "PRODUCT"]]),
            "validation": Dataset.from_pandas(valid[["input", "PRODUCT"]]),
        }
    )

    # load tokenizer
    try:  # load pretrained tokenizer from local directory
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.abspath(CFG.pretrained_model_name_or_path), return_tensors="pt"
        )
    except:  # load pretrained tokenizer from huggingface model hub
        tokenizer = AutoTokenizer.from_pretrained(
            CFG.pretrained_model_name_or_path, return_tensors="pt"
        )
    tokenizer.add_tokens(
        [
            ".",
            "6",
            "7",
            "8",
            "<",
            ">",
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "Ba",
            "Bi",
            "Ca",
            "Cl",
            "Cu",
            "Fe",
            "Ge",
            "Hg",
            "K",
            "Li",
            "Mg",
            "Mn",
            "Mo",
            "Na",
            "Nd",
            "Ni",
            "P",
            "Pb",
            "Pd",
            "Pt",
            "Re",
            "Rh",
            "Ru",
            "Ru",
            "Sb",
            "Si",
            "Sm",
            "Ta",
            "Ti",
            "Tl",
            "W",
            "Yb",
            "Zn",
            "Zr",
            "e",
            "p",
        ]
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": tokenizer.additional_special_tokens
            + ["REACTANT:", "REAGENT:"]
        }
    )
    CFG.tokenizer = tokenizer

    # load model
    try:  # load pretrained model from local directory
        model = AutoModelForSeq2SeqLM.from_pretrained(
            os.path.abspath(CFG.pretrained_model_name_or_path)
        )
    except:  # load pretrained model from huggingface model hub
        model = AutoModelForSeq2SeqLM.from_pretrained(CFG.pretrained_model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = dataset.map(
        lambda examples: preprocess_dataset(examples, CFG),
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        CFG.model,
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

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model("./best_model")
