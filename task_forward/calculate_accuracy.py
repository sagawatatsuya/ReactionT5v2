import argparse
import os
import sys
import warnings

import pandas as pd
import rdkit
from rdkit import Chem
from transformers import AutoTokenizer

rdkit.RDLogger.DisableLog("rdApp.*")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import canonicalize, seed_everything

warnings.filterwarnings("ignore")

"""
Score forward-reaction (product) predictions produced by prediction.py.

Expected columns in --input_data: `input`, `0th` ... `{num_beams-1}th`.
The target column is read from --target_data via --target_col.
Reports top-k accuracies (k=1,2,3,5) and invalid SMILES rate.
"""


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compute accuracy for forward reaction product predictions."
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the input data.",
    )
    parser.add_argument(
        "--target_data",
        type=str,
        required=True,
        help="Path to the target data.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        required=True,
        help="Name of target column.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="sagawa/ReactionT5v2-forward",
        help="Finetuned model path/name (only used to load tokenizer).",
    )
    parser.add_argument(
        "--num_beams", type=int, default=5, help="Number of beams used for beam search."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility."
    )
    return parser.parse_args()


def remove_space(row, num_beams: int):
    """Strip whitespace inside predicted SMILES strings."""
    for i in range(num_beams):
        col = f"{i}th"
        row[col] = row[col].replace(" ", "")
    return row


if __name__ == "__main__":
    CFG = parse_args()

    seed_everything(seed=CFG.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.abspath(CFG.model_name_or_path)
        if os.path.exists(CFG.model_name_or_path)
        else CFG.model_name_or_path,
        return_tensors="pt",
    )

    df = pd.read_csv(CFG.input_data)
    pred_cols = [f"{i}th" for i in range(CFG.num_beams)]
    df[pred_cols] = df[pred_cols].fillna(" ")
    df["target"] = pd.read_csv(CFG.target_data)[CFG.target_col].values
    df = df.apply(lambda row: remove_space(row, CFG.num_beams), axis=1)

    top_metrics = {k: [] for k in (1, 2, 3, 5)}
    invalidity = []

    for _, row in df.iterrows():
        target = canonicalize(row["target"])
        preds = [canonicalize(row[col]) for col in pred_cols]

        match_idx = next((i for i, pred in enumerate(preds) if pred == target), None)
        for k in top_metrics:
            top_metrics[k].append(1 if match_idx is not None and match_idx < k else 0)

        invalidity.append(
            sum(Chem.MolFromSmiles(row[col].rstrip(".")) is None for col in pred_cols)
        )

    print(CFG.input_data)
    for k, values in top_metrics.items():
        print(f"Top {k} accuracy: {sum(values) / len(values):.4f}")
    print(
        f"Top {CFG.num_beams} invalidity: {sum(invalidity) / (len(invalidity) * CFG.num_beams) * 100:.2f}"
    )
