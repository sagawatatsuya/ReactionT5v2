import os
import sys
import subprocess
import warnings
import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer
from datasets.utils.logging import disable_progress_bar

# Append the utils module path
sys.path.append("../")
from utils import seed_everything, canonicalize, get_logger
from train import train_loop

# Suppress warnings and logging
warnings.filterwarnings("ignore")
disable_progress_bar()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training script for ReactionT5Yield model."
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to training data CSV file.",
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        required=True,
        help="Path to validation data CSV file.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sagawa/CompoundT5",
        help="Pretrained model name or path.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The model's name or path used for fine-tuning.",
    )
    parser.add_argument(
        "--download_pretrained_model",
        action="store_true",
        default=False,
        required=False,
        help="Download pretrained model from hugging face hub and use it for fine-tuning.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience."
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--max_len", type=int, default=300, help="Maximum input token length."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers."
    )
    parser.add_argument(
        "--fc_dropout",
        type=float,
        default=0.0,
        help="Dropout rate after fully connected layers.",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-6, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=int,
        default=1000,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of warmup steps."
    )
    parser.add_argument(
        "--batch_scheduler", action="store_true", help="Use batch scheduler."
    )
    parser.add_argument(
        "--print_freq", type=int, default=100, help="Logging frequency."
    )
    parser.add_argument(
        "--use_apex", action="store_true", help="Use apex for mixed precision training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--sampling_num",
        type=int,
        default=-1,
        help="Number of samples used for training. If you want to use all samples, set -1.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint file for resuming training.",
    )

    return parser.parse_args()


def preprocess(df, cfg):
    """
    Preprocess the input DataFrame for training.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cfg (argparse.Namespace): Configuration object.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    for col in ["REAGENT", "REACTANT", "PRODUCT"]:
        df[col] = df[col].apply(lambda x: canonicalize(x) if x != " " else " ")
    if "YIELD" in df.columns:
        df["YIELD"] = df["YIELD"].clip(0, 100) / 100
    else:
        df["YIELD"] = None
    df["input"] = (
        "REACTANT:"
        + df["REACTANT"]
        + "REAGENT:"
        + df["REAGENT"]
        + "PRODUCT:"
        + df["PRODUCT"]
    )
    df = df[["input", "YIELD"]].drop_duplicates().reset_index(drop=True)

    if cfg.debug:
        df = df.head(1000)

    return df


def download_pretrained_model():
    """
    Download the pretrained model from Hugging Face.
    """
    subprocess.run(
        "wget https://huggingface.co/sagawa/ReactionT5v2-yield/resolve/main/CompoundT5_best.pth",
        shell=True,
    )
    subprocess.run(
        "wget https://huggingface.co/sagawa/ReactionT5v2-yield/resolve/main/config.pth",
        shell=True,
    )
    subprocess.run(
        "wget https://huggingface.co/sagawa/ReactionT5v2-yield/resolve/main/special_tokens_map.json",
        shell=True,
    )
    subprocess.run(
        "wget hhttps://huggingface.co/sagawa/ReactionT5v2-yield/resolve/main/tokenizer.json",
        shell=True,
    )
    subprocess.run(
        "wget https://huggingface.co/sagawa/ReactionT5v2-yield/resolve/main/tokenizer_config.json",
        shell=True,
    )


if __name__ == "__main__":
    CFG = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG.device = device
    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)
    seed_everything(seed=CFG.seed)

    if CFG.download_pretrained_model:
        download_pretrained_model()
        CFG.model_name_or_path = "."

    train = pd.read_csv(CFG.train_data_path).drop_duplicates().reset_index(drop=True)
    valid = pd.read_csv(CFG.valid_data_path).drop_duplicates().reset_index(drop=True)
    train = preprocess(train, CFG)
    valid = preprocess(valid, CFG)

    if CFG.sampling_num > 0:
        train = train.sample(n=CFG.sampling_num, random_state=CFG.seed).reset_index(
            drop=True
        )

    LOGGER = get_logger(os.path.join(CFG.output_dir, "train"))
    CFG.logger = LOGGER

    try:  # load pretrained tokenizer from local directory
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.abspath(CFG.model_name_or_path), return_tensors="pt"
        )
    except:  # load pretrained tokenizer from huggingface model hub
        tokenizer = AutoTokenizer.from_pretrained(
            CFG.model_name_or_path, return_tensors="pt"
        )
    tokenizer.save_pretrained(CFG.output_dir)
    CFG.tokenizer = tokenizer

    train_loop(train, valid, CFG)
