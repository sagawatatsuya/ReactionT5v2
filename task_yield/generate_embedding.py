import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, T5EncoderModel
import argparse
from torch.utils.data import DataLoader
import numpy as np
sys.path.append("../")
from utils import seed_everything, filter_out
from generation_utils import ReactionT5Dataset
from train import preprocess_df


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
        "--model_name_or_path",
        type=str,
        default="sagawa/ReactionT5v2-yield",
        help="Name or path of the finetuned model for prediction. Can be a local model or one from Hugging Face.",
    )
    parser.add_argument("--debug", action="store_true", help="Use debug mode.")
    parser.add_argument(
        "--input_max_length", type=int, default=400, help="Maximum token length of input."
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
    outputs_mean = []
    model.eval()
    model.to(device)
    for inputs in dataloader:
        inputs = {k: v.to(CFG.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        outputs_mean.append(output[0].mean(dim=1).detach().cpu().numpy())

    return outputs_mean


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

    model = T5EncoderModel.from_pretrained(CFG.model_name_or_path).to(CFG.device)
    model.eval()

    input_data = filter_out(pd.read_csv(CFG.input_data), ["YIELD", "REACTANT", "PRODUCT"])
    input_data.to_csv(os.path.join(CFG.output_dir, "input_data.csv"), index=False)
    input_data = preprocess_df(input_data, CFG, drop_duplicates=False)
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

    np.save(os.path.join(CFG.output_dir, "embedding_mean.npy"), outputs)