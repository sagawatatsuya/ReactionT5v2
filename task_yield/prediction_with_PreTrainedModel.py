import argparse
import logging
import os
import sys
import warnings

import pandas as pd
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
disable_progress_bar()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Append the utils module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation_utils import ReactionT5Dataset
from models import ReactionT5Yield2
from prediction import inference_fn
from train import preprocess_df
from utils import seed_everything


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prediction script for ReactionT5Yield model."
    )

    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Data as a CSV file that contains an 'input' column. The format of the contents of the column are like 'REACTANT:{reactants of the reaction}PRODUCT:{products of the reaction}'. If there are multiple reactants, concatenate them with '.'.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="sagawa/ReactionT5v2-yield",
        help="Name or path of the finetuned model for prediction. Can be a local model or one from Hugging Face.",
    )
    parser.add_argument("--debug", action="store_true", help="Use debug mode.")
    parser.add_argument(
        "--input_max_length",
        type=int,
        default=400,
        help="Maximum token length of input.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, required=False, help="Batch size."
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
        "--output_dir",
        type=str,
        default="./",
        help="Directory where predictions are saved.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    return parser.parse_args()


if __name__ == "__main__":
    CFG = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG.device = device

    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

    seed_everything(seed=CFG.seed)

    CFG.tokenizer = AutoTokenizer.from_pretrained(
        os.path.abspath(CFG.model_name_or_path)
        if os.path.exists(CFG.model_name_or_path)
        else CFG.model_name_or_path,
        return_tensors="pt",
    )

    model = ReactionT5Yield2.from_pretrained(CFG.model_name_or_path)

    test_ds = pd.read_csv(CFG.input_data)
    test_ds = preprocess_df(test_ds, CFG, drop_duplicates=False)

    test_dataset = ReactionT5Dataset(CFG, test_ds)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    prediction = inference_fn(test_loader, model, CFG)

    test_ds["prediction"] = prediction
    test_ds["prediction"] = test_ds["prediction"].clip(0, 100)
    test_ds.to_csv(
        os.path.join(CFG.output_dir, "yield_prediction_output.csv"), index=False
    )
