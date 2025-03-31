import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation_utils import ReactionT5Dataset
from models import ReactionT5Yield2
from train import preprocess_df
from utils import filter_out, seed_everything


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
        help="Data as a string or CSV file that contains an 'input' column. The format of the string or contents of the column are like 'REACTANT:{reactants of the reaction}PRODUCT:{products of the reaction}'. If there are multiple reactants, concatenate them with '.'.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=False,
        help="Path to the test data. If provided, the duplicates will be removed from the input data.",
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


def create_embedding(dataloader, model, device):
    outputs = []
    model.eval()
    model.to(device)
    for inputs in dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate_embedding(inputs)

        outputs.append(output.detach().cpu().numpy())

    return outputs


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

    model = ReactionT5Yield2.from_pretrained(CFG.model_name_or_path).to(CFG.device)
    model.eval()

    input_data = filter_out(
        pd.read_csv(CFG.input_data), ["YIELD", "REACTANT", "PRODUCT"]
    )
    input_data = preprocess_df(input_data, CFG, drop_duplicates=False)
    if CFG.test_data:
        test_data = filter_out(
            pd.read_csv(CFG.test_data), ["YIELD", "REACTANT", "PRODUCT"]
        )
        test_data = preprocess_df(test_data, CFG, drop_duplicates=False)
        # Remove duplicates from the input data
        input_data = input_data[
            ~input_data["input"].isin(test_data["input"])
        ].reset_index(drop=True)
    input_data.to_csv(os.path.join(CFG.output_dir, "input_data.csv"), index=False)
    dataset = ReactionT5Dataset(CFG, input_data)
    dataloader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    outputs = create_embedding(dataloader, model, CFG.device)
    outputs = np.concatenate(outputs, axis=0)

    np.save(os.path.join(CFG.output_dir, "embedding_mean_v2.npy"), outputs)
