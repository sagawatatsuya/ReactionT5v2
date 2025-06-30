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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for reaction retrosynthesis prediction."
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
        default="sagawa/ReactionT5v2-retrosynthesis",
        help="Name or path of the finetuned model for prediction. Can be a local model or one from Hugging Face.",
    )
    parser.add_argument(
        "--num_beams", type=int, default=5, help="Number of beams used for beam search."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility."
    )
    return parser.parse_args()


def remove_space(row):
    for i in range(5):
        row[f"{i}th"] = row[f"{i}th"].replace(" ", "")
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
    df[[f"{i}th" for i in range(CFG.num_beams)]] = df[
        [f"{i}th" for i in range(CFG.num_beams)]
    ].fillna(" ")
    df["target"] = pd.read_csv(CFG.target_data)[CFG.target_col].values
    df = df.apply(remove_space, axis=1)

    top_k_invalidity = CFG.num_beams

    top1, top2, top3, top5 = [], [], [], []
    invalidity = []

    for idx, row in df.iterrows():
        target = canonicalize(row["target"])
        if canonicalize(row["0th"]) == target:
            top1.append(1)
            top2.append(1)
            top3.append(1)
            top5.append(1)
        elif canonicalize(row["1th"]) == target:
            top1.append(0)
            top2.append(1)
            top3.append(1)
            top5.append(1)
        elif canonicalize(row["2th"]) == target:
            top1.append(0)
            top2.append(0)
            top3.append(1)
            top5.append(1)
        elif canonicalize(row["3th"]) == target:
            top1.append(0)
            top2.append(0)
            top3.append(0)
            top5.append(1)
        elif canonicalize(row["4th"]) == target:
            top1.append(0)
            top2.append(0)
            top3.append(0)
            top5.append(1)
        else:
            top1.append(0)
            top2.append(0)
            top3.append(0)
            top5.append(0)

        input_compound = row["input"]
        output = [row[f"{i}th"] for i in range(top_k_invalidity)]
        inval_score = 0
        for ith, out in enumerate(output):
            mol = Chem.MolFromSmiles(out.rstrip("."))
            if not isinstance(mol, Chem.rdchem.Mol):
                inval_score += 1
        invalidity.append(inval_score)
    print(CFG.input_data)
    print(f"Top 1 accuracy: {sum(top1) / len(top1)}")
    print(f"Top 2 accuracy: {sum(top2) / len(top2)}")
    print(f"Top 3 accuracy: {sum(top3) / len(top3)}")
    print(f"Top 5 accuracy: {sum(top5) / len(top5)}")
    print(
        f"Top {top_k_invalidity} Invalidity: {sum(invalidity) / (len(invalidity) * top_k_invalidity) * 100}"
    )
