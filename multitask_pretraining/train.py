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
from utils import (
    seed_everything,
    get_accuracy_score_multitask,
    canonicalize,
    space_clean,
    preprocess_dataset,
    filter_out,
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
        "--train_data_path_FORWARD",
        type=str,
        required=True,
        help="Path to training data CSV.",
    )
    parser.add_argument(
        "--valid_data_path_FORWARD",
        type=str,
        required=True,
        help="Path to validation data CSV.",
    )
    parser.add_argument(
        "--test_data_path_FORWARD", type=str, help="Path to test data CSV."
    )
    parser.add_argument(
        "--train_data_path_RETROSYNTHESIS",
        type=str,
        required=True,
        help="Path to training data CSV.",
    )
    parser.add_argument(
        "--valid_data_path_RETROSYNTHESIS",
        type=str,
        required=True,
        help="Path to validation data CSV.",
    )
    parser.add_argument(
        "--test_data_path_RETROSYNTHESIS", type=str, help="Path to test data CSV."
    )
    parser.add_argument(
        "--train_data_path_YIELD",
        type=str,
        required=True,
        help="Path to training data CSV.",
    )
    parser.add_argument(
        "--valid_data_path_YIELD",
        type=str,
        required=True,
        help="Path to validation data CSV.",
    )
    parser.add_argument(
        "--test_data_path_YIELD", type=str, help="Path to test data CSV."
    )
    parser.add_argument(
        "--USPTO_MIT_test_data_path", type=str, help="Path to USPTO_MIT test data CSV."
    )
    parser.add_argument(
        "--USPTO_50k_test_data_path", type=str, help="Path to USPTO_50k test data CSV."
    )
    parser.add_argument(
        "--CN_test_data_path", type=str, help="Path to CN test data CSV."
    )
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
        default="target",
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


def preprocess_df_FORWARD(df):
    """Preprocess the dataframe by filling NaNs, dropping duplicates, and formatting the input."""
    for col in ["CATALYST", "REAGENT", "SOLVENT"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].fillna(" ")

    df = (
        df[["REACTANT", "PRODUCT", "CATALYST", "REAGENT", "SOLVENT"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df["REAGENT"] = df["CATALYST"] + "." + df["REAGENT"] + "." + df["SOLVENT"]
    df["REAGENT"] = df["REAGENT"].apply(lambda x: space_clean(x))
    df["REAGENT"] = df["REAGENT"].apply(lambda x: canonicalize(x) if x != " " else " ")
    df["input"] = (
        "TASK_FORWARD" + "REACTANT:" + df["REACTANT"] + "REAGENT:" + df["REAGENT"]
    )
    df["target"] = df["PRODUCT"]

    return df


def preprocess_USPTO_MIT(df):
    df["REACTANT"] = df["REACTANT"].apply(lambda x: str(sorted(x.split("."))))
    df["REAGENT"] = df["REAGENT"].apply(lambda x: str(sorted(x.split("."))))
    df["PRODUCT"] = df["PRODUCT"].apply(lambda x: str(sorted(x.split("."))))

    df["input"] = "REACTANT:" + df["REACTANT"] + "REAGENT:" + df["REAGENT"]
    df["pair"] = df["input"] + " - " + df["PRODUCT"].astype(str)

    return df


def preprocess_df_RETROSYNTHESIS(df):
    """Preprocess the dataframe by filling NaNs, dropping duplicates, and formatting the input."""
    df = df.drop_duplicates(subset=["REACTANT", "PRODUCT"]).reset_index(drop=True)
    df["input"] = "TASK_RETROSYNTHESIS" + "PRODUCT:" + df["PRODUCT"]
    df["target"] = df["REACTANT"]

    return df


def preprocess_USPTO_50k(df):
    df["REACTANT"] = df["REACTANT"].apply(lambda x: str(sorted(x.split("."))))
    df["PRODUCT"] = df["PRODUCT"].apply(lambda x: str(sorted(x.split("."))))

    df["pair"] = df["REACTANT"] + " - " + df["PRODUCT"].astype(str)

    return df


def preprocess_df_YIELD(df):
    for col in ["CATALYST", "REAGENT", "SOLVENT"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].fillna(" ")
    df["REAGENT"] = df["CATALYST"] + "." + df["REAGENT"]
    df["REAGENT"] = df["REAGENT"].apply(lambda x: space_clean(x))
    df["REAGENT"] = df["REAGENT"].apply(lambda x: canonicalize(x) if x != " " else " ")
    df = df.drop_duplicates(subset=["REACTANT", "REAGENT", "PRODUCT"]).reset_index(
        drop=True
    )
    if max(df["YIELD"]) > 1:
        df["YIELD"] = df["YIELD"] / 100
    df["YIELD"] = df["YIELD"].apply(lambda x: round(x * 10) / 10)
    # convert YIELD value 0.1 to 10%, 0.2 to 20%, ..., 1.0 to 100%
    df["YIELD"] = df["YIELD"].apply(lambda x: str(int(x * 100)) + "%")

    df["input"] = (
        "TASK_YIELD"
        + "REACTANT:"
        + df["REACTANT"]
        + "REAGENT:"
        + df["REAGENT"]
        + "PRODUCT:"
        + df["PRODUCT"]
    )

    df["target"] = df["YIELD"]

    return df


def preprocess_CN(df):
    """
    Preprocess the CN test DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df["REACTANT"] = df["REACTANT"].apply(lambda x: ".".join(sorted(x.split("."))))
    df["REAGENT"] = df["REAGENT"].apply(lambda x: ".".join(sorted(x.split("."))))
    df["PRODUCT"] = df["PRODUCT"].apply(lambda x: ".".join(sorted(x.split("."))))
    df["input"] = (
        "REACTANT:"
        + df["REACTANT"]
        + "REAGENT:"
        + df["REAGENT"]
        + "PRODUCT:"
        + df["PRODUCT"]
    )
    df["pair"] = df["input"]
    return df


if __name__ == "__main__":
    CFG = parse_args()
    CFG.disable_tqdm = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=CFG.seed)

    train_dfs = []
    valid_dfs = []

    # FORWARD
    train = preprocess_df_FORWARD(
        filter_out(pd.read_csv(CFG.train_data_path_FORWARD), ["REACTANT", "PRODUCT"])
    )
    valid = preprocess_df_FORWARD(
        filter_out(pd.read_csv(CFG.valid_data_path_FORWARD), ["REACTANT", "PRODUCT"])
    )
    if CFG.USPTO_MIT_test_data_path:
        train_copy = preprocess_USPTO_MIT(train.copy())
        USPTO_test = preprocess_USPTO_MIT(pd.read_csv(CFG.USPTO_MIT_test_data_path))
        train = train[~train_copy["pair"].isin(USPTO_test["pair"])].reset_index(
            drop=True
        )
    train["pair"] = train["input"] + " - " + train["target"]
    valid["pair"] = valid["input"] + " - " + valid["target"]
    valid = valid[~valid["pair"].isin(train["pair"])].reset_index(drop=True)

    train.to_csv("train_FORWARD.csv", index=False)
    valid.to_csv("valid_FORWARD.csv", index=False)
    train_dfs.append(train[["input", "target"]])
    valid_dfs.append(valid[["input", "target"]])

    if CFG.test_data_path_FORWARD:
        test = preprocess_df_FORWARD(
            filter_out(pd.read_csv(CFG.test_data_path_FORWARD), ["REACTANT", "PRODUCT"])
        )
        test["pair"] = test["input"] + " - " + test["target"]
        test = test[~test["pair"].isin(train["pair"])].reset_index(drop=True)
        test.to_csv("test_FORWARD.csv", index=False)

    # RETROSYNTHESIS
    train = preprocess_df_RETROSYNTHESIS(
        filter_out(
            pd.read_csv(CFG.train_data_path_RETROSYNTHESIS), ["REACTANT", "PRODUCT"]
        )
    )
    valid = preprocess_df_RETROSYNTHESIS(
        filter_out(
            pd.read_csv(CFG.valid_data_path_RETROSYNTHESIS), ["REACTANT", "PRODUCT"]
        )
    )
    if CFG.USPTO_50k_test_data_path:
        train_copy = preprocess_USPTO_50k(train.copy())
        USPTO_test = preprocess_USPTO_50k(pd.read_csv(CFG.USPTO_50k_test_data_path))
        train = train[~train_copy["pair"].isin(USPTO_test["pair"])].reset_index(
            drop=True
        )
    train["pair"] = train["input"] + " - " + train["target"]
    valid["pair"] = valid["input"] + " - " + valid["target"]
    valid = valid[~valid["pair"].isin(train["pair"])].reset_index(drop=True)

    train.to_csv("train_RETROSYNTHESIS.csv", index=False)
    valid.to_csv("valid_RETROSYNTHESIS.csv", index=False)
    train_dfs.append(train[["input", "target"]])
    valid_dfs.append(valid[["input", "target"]])

    if CFG.test_data_path_RETROSYNTHESIS:
        test = preprocess_df_RETROSYNTHESIS(
            filter_out(
                pd.read_csv(CFG.test_data_path_RETROSYNTHESIS), ["REACTANT", "PRODUCT"]
            )
        )
        test["pair"] = test["input"] + " - " + test["target"]
        test = test[~test["pair"].isin(train["pair"])].reset_index(drop=True)
        test.to_csv("test_RETROSYNTHESIS.csv", index=False)

    # YIELD
    train = preprocess_df_YIELD(
        filter_out(
            pd.read_csv(CFG.train_data_path_YIELD), ["REACTANT", "PRODUCT", "YIELD"]
        )
    )
    valid = preprocess_df_YIELD(
        filter_out(
            pd.read_csv(CFG.valid_data_path_YIELD), ["REACTANT", "PRODUCT", "YIELD"]
        )
    )
    if CFG.CN_test_data_path:
        train_copy = preprocess_CN(train.copy())
        CN_test = preprocess_CN(pd.read_csv(CFG.CN_test_data_path))
        train = train[~train_copy["pair"].isin(CN_test["pair"])].reset_index(drop=True)
    train["pair"] = train["input"] + " - " + train["target"].astype(str)
    valid["pair"] = valid["input"] + " - " + valid["target"].astype(str)
    valid = valid[~valid["pair"].isin(train["pair"])].reset_index(drop=True)

    train.to_csv("train_YIELD.csv", index=False)
    valid.to_csv("valid_YIELD.csv", index=False)
    train_dfs.append(train[["input", "target"]])
    valid_dfs.append(valid[["input", "target"]])

    if CFG.test_data_path_YIELD:
        test = preprocess_df_YIELD(
            filter_out(
                pd.read_csv(CFG.test_data_path_YIELD), ["REACTANT", "PRODUCT", "YIELD"]
            )
        )
        test["pair"] = test["input"] + " - " + test["target"].astype(str)
        test = test[~test["pair"].isin(train["pair"])].reset_index(drop=True)
        test.to_csv("test_YIELD.csv", index=False)

    train = pd.concat(train_dfs, axis=0).reset_index(drop=True)
    valid = pd.concat(valid_dfs, axis=0).reset_index(drop=True)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train[["input", CFG.target_column]]),
            "validation": Dataset.from_pandas(valid[["input", CFG.target_column]]),
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
            + [
                "REACTANT:",
                "REAGENT:",
                "PRODUCT:",
                "0%",
                "10%",
                "20%",
                "30%",
                "40%",
                "50%",
                "60%",
                "70%",
                "80%",
                "90%",
                "100%",
                "TASK_FORWARD",
                "TASK_RETROSYNTHESIS",
                "TASK_YIELD",
            ]
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

    model.config.eval_beams = CFG.eval_beams
    model.config.max_length = CFG.target_max_length
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: get_accuracy_score_multitask(
            eval_preds, CFG
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    try:
        trainer.train(resume_from_checkpoint=True)
    except:
        trainer.train(resume_from_checkpoint=None)
    trainer.save_model("./best_model")
