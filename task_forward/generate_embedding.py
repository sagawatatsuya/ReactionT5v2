import os
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, T5EncoderModel
import argparse
from torch.utils.data import DataLoader
import sys
import numpy as np

sys.path.append("../")
from utils import seed_everything
from prediction import ProductDataset

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for reaction product prediction."
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the input data.",
    )
    parser.add_argument(
        "--input_column",
        type=str,
        default="input",
        help="Column name used for model input.",
    )
    parser.add_argument(
        "--input_max_length",
        type=int,
        default=400,
        help="Maximum token length of input.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="sagawa/ReactionT5v2-forward",
        help="Name or path of the finetuned model for prediction. Can be a local model or one from Hugging Face.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for prediction."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory where predictions are saved.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Use debug mode."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility."
    )
    return parser.parse_args()


def create_embedding(dataloader, model, device):
    outputs = []
    outputs_cls = []
    model.eval()
    model.to(device)
    for inputs in dataloader:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            output = model(**inputs)
        outputs.append(output[0].detach().cpu().numpy())
        outputs_cls.append(output[0][:, 0, :].detach().cpu().numpy())

    return outputs, outputs_cls


if __name__ == "__main__":
    CFG = parse_args()
    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

    seed_everything(seed=CFG.seed)

    CFG.tokenizer = AutoTokenizer.from_pretrained(
        CFG.model_name_or_path, return_tensors="pt"
    )
    model = T5EncoderModel.from_pretrained(CFG.model_name_or_path).to(CFG.device)
    model.eval()

    input_data = pd.read_csv(CFG.input_data)
    dataset = ProductDataset(CFG, input_data)
    dataloader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    outputs, outputs_cls = create_embedding(dataloader, model, CFG.device)

    outputs = np.concatenate(outputs, axis=0)
    outputs_cls = np.concatenate(outputs_cls, axis=0)

    np.save(os.path.join(CFG.output_dir, "embedding.npy"), outputs)
    np.save(os.path.join(CFG.output_dir, "embedding_cls.npy"), outputs_cls)
