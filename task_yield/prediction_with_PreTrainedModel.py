import os
import warnings
import logging
import sys
import argparse

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets.utils.logging import disable_progress_bar

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
disable_progress_bar()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Append the utils module path
sys.path.append("../")
from utils import seed_everything
from train import prepare_input, inference_fn, preprocess_df
from models import ReactionT5Yield2


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prediction script for ReactionT5Yield model."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Data as a string or CSV file that contains an 'input' column. The format of the string or contents of the column are like 'REACTANT:{reactants of the reaction}PRODUCT:{products of the reaction}'. If there are multiple reactants, concatenate them with '.'.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The name of a finetuned model or path to a model which you want to use for prediction. You can use your local models or models uploaded to hugging face.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--input_max_length", type=int, default=400, help="Maximum input token length."
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
        help="Directory to save the prediction.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    return parser.parse_args()


class TestDataset(Dataset):
    """
    Dataset class for training.
    """

    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df["input"].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.inputs[item])

        return inputs


if __name__ == "__main__":
    CFG = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG.device = device

    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

    seed_everything(seed=CFG.seed)

    CFG.tokenizer = AutoTokenizer.from_pretrained(
        CFG.model_name_or_path, return_tensors="pt"
    )

    model = ReactionT5Yield2.from_pretrained(CFG.model_name_or_path)

    if CFG.data.endswith(".csv"):
        test_ds = pd.read_csv(CFG.data)
        if "input" not in test_ds.columns:
            test_ds = preprocess_df(test_ds, CFG)
    else:
        test_ds = pd.DataFrame.from_dict({"input": [CFG.data]}, orient="index").T

    test_dataset = TestDataset(CFG, test_ds)
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
    test_ds.to_csv(os.path.join(CFG.output_dir, 'yield_prediction_output.csv'), index=False)
