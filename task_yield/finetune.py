import argparse
import os
import subprocess
import sys
import warnings

import pandas as pd
import torch
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train import preprocess_df, train_loop
from utils import get_logger, seed_everything

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
        "--similar_reaction_data_path",
        type=str,
        required=False,
        help="Path to similar data CSV.",
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
        "--input_max_length", type=int, default=300, help="Maximum input token length."
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
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision for training.",
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
        "--sampling_frac",
        type=float,
        default=-1.0,
        help="Ratio of samples used for training. If you want to use all samples, set -1.0.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint file for resuming training.",
    )

    return parser.parse_args()


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
        "wget https://huggingface.co/sagawa/ReactionT5v2-yield/resolve/main/tokenizer.json",
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
    train = preprocess_df(train, CFG)
    valid = preprocess_df(valid, CFG)

    if CFG.sampling_num > 0:
        train = train.sample(n=CFG.sampling_num, random_state=CFG.seed).reset_index(
            drop=True
        )
    elif CFG.sampling_frac > 0 and CFG.sampling_frac < 1:
        train = train.sample(frac=CFG.sampling_frac, random_state=CFG.seed).reset_index(
            drop=True
        )

    if CFG.similar_reaction_data_path:
        similar = preprocess_df(pd.read_csv(CFG.similar_reaction_data_path), CFG)
        print(len(train))
        train = pd.concat([train, similar], ignore_index=True)
        print(len(train))

    LOGGER = get_logger(os.path.join(CFG.output_dir, "train"))
    CFG.logger = LOGGER

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.abspath(CFG.model_name_or_path)
        if os.path.exists(CFG.model_name_or_path)
        else CFG.model_name_or_path,
        return_tensors="pt",
    )
    tokenizer.save_pretrained(CFG.output_dir)
    CFG.tokenizer = tokenizer

    train_loop(train, valid, CFG)
