import argparse
import os
import sys
import warnings
from pathlib import Path

import datasets
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import (
    add_new_tokens,
    filter_out,
    get_accuracy_score,
    preprocess_dataset,
    seed_everything,
)

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
    parser.add_argument(
        "--USPTO_test_data_path",
        type=str,
        help="The path to data used for USPTO testing. CSV file that contains ['REACTANT', 'PRODUCT'] columns is expected.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="t5", help="Path of the output directory."
    )
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
        "--input_max_length",
        type=int,
        default=400,
        help="Max input token length.",
    )
    parser.add_argument(
        "--target_max_length",
        type=int,
        default=150,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def preprocess_df(df, drop_duplicates=True):
    """Preprocess the dataframe by filling NaNs, dropping duplicates, and formatting the input."""
    for col in ["REACTANT", "PRODUCT", "CATALYST", "REAGENT", "SOLVENT"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].fillna(" ")

    if drop_duplicates:
        df = (
            df[["REACTANT", "PRODUCT", "CATALYST", "REAGENT", "SOLVENT"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
    df["input"] = df["PRODUCT"]

    return df


def preprocess_USPTO(df):
    df["REACTANT"] = df["REACTANT"].apply(lambda x: str(sorted(x.split("."))))
    df["PRODUCT"] = df["PRODUCT"].apply(lambda x: str(sorted(x.split("."))))

    df["pair"] = df["REACTANT"] + " - " + df["PRODUCT"].astype(str)

    return df


if __name__ == "__main__":
    CFG = parse_args()
    CFG.disable_tqdm = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=CFG.seed)

    train = preprocess_df(
        filter_out(pd.read_csv(CFG.train_data_path), ["REACTANT", "PRODUCT"])
    )
    valid = preprocess_df(
        filter_out(pd.read_csv(CFG.valid_data_path), ["REACTANT", "PRODUCT"])
    )
    if CFG.USPTO_test_data_path:
        train_copy = preprocess_USPTO(train.copy())
        USPTO_test = preprocess_USPTO(pd.read_csv(CFG.USPTO_test_data_path))
        train = train[~train_copy["pair"].isin(USPTO_test["pair"])].reset_index(
            drop=True
        )
    train["pair"] = train["REACTANT"] + " - " + train["PRODUCT"]
    valid["pair"] = valid["REACTANT"] + " - " + valid["PRODUCT"]
    valid = valid[~valid["pair"].isin(train["pair"])].reset_index(drop=True)
    train.to_csv("train.csv", index=False)
    valid.to_csv("valid.csv", index=False)

    if CFG.test_data_path:
        test = preprocess_df(
            filter_out(pd.read_csv(CFG.test_data_path), ["REACTANT", "PRODUCT"])
        )
        test["pair"] = test["REACTANT"] + " - " + test["PRODUCT"]
        test = test[~test["pair"].isin(train["pair"])].reset_index(drop=True)
        test = test.drop_duplicates(subset=["pair"]).reset_index(drop=True)
        test.to_csv("test.csv", index=False)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train[["input", "REACTANT"]]),
            "validation": Dataset.from_pandas(valid[["input", "REACTANT"]]),
        }
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.abspath(CFG.pretrained_model_name_or_path)
        if os.path.exists(CFG.pretrained_model_name_or_path)
        else CFG.pretrained_model_name_or_path,
        return_tensors="pt",
    )
    tokenizer = add_new_tokens(
        tokenizer,
        Path(__file__).resolve().parent.parent / "data" / "additional_tokens.txt",
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": tokenizer.additional_special_tokens
            + ["REACTANT:", "REAGENT:"]
        }
    )
    CFG.tokenizer = tokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(
        os.path.abspath(CFG.pretrained_model_name_or_path) if os.path.exists(CFG.pretrained_model_name_or_path) else CFG.pretrained_model_name_or_path
    )
    model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = dataset.map(
        lambda examples: preprocess_dataset(examples, CFG),
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        CFG.output_dir,
        eval_strategy=CFG.evaluation_strategy,
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
        metric_for_best_model="accuracy",
        greater_is_better=True
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

    try:
        trainer.train(resume_from_checkpoint=True)
    except:
        trainer.train(resume_from_checkpoint=None)
    trainer.save_model("./best_model")
