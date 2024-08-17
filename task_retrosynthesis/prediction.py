import os
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
from torch.utils.data import Dataset, DataLoader
import sys
import gc

sys.path.append("../")
from utils import seed_everything

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
        "--output_min_length",
        type=int,
        default=1,
        help="Minimum token length of output.",
    )
    parser.add_argument(
        "--output_max_length",
        type=int,
        default=300,
        help="Maximum token length of output.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="sagawa/ReactionT5-product-prediction",
        help="Name or path of the finetuned model for prediction. Can be a local model or one from Hugging Face.",
    )
    parser.add_argument(
        "--num_beams", type=int, default=5, help="Number of beams used for beam search."
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=5,
        help="Number of predictions returned. Must be less than or equal to num_beams.",
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


def prepare_input(cfg, text):
    inputs = cfg.tokenizer(
        text,
        return_tensors="pt",
        max_length=cfg.input_max_length,
        padding="max_length",
        truncation=True,
    )
    dic = {"input_ids": [], "attention_mask": []}
    for k, v in inputs.items():
        dic[k].append(torch.tensor(v[0], dtype=torch.long))
    return dic


class ProductDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df[cfg.input_column].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return prepare_input(self.cfg, self.inputs[idx])


def predict_single_input(input_compound, model, cfg):
    inp = cfg.tokenizer(input_compound, return_tensors="pt").to(cfg.device)
    with torch.no_grad():
        output = model.generate(
            **inp,
            min_length=cfg.output_min_length,
            max_length=cfg.output_max_length,
            num_beams=cfg.num_beams,
            num_return_sequences=cfg.num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return output


def decode_output(output, cfg):
    sequences = [
        cfg.tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "").rstrip(".")
        for seq in output["sequences"]
    ]
    if cfg.num_beams > 1:
        scores = output["sequences_scores"].tolist()
        return sequences, scores
    return sequences, None


def save_single_prediction(input_compound, output, scores, cfg):
    output_data = [input_compound] + output + (scores if scores else [])
    columns = (
        ["input"]
        + [f"{i}th" for i in range(cfg.num_beams)]
        + ([f"{i}th score" for i in range(cfg.num_beams)] if scores else [])
    )
    output_df = pd.DataFrame([output_data], columns=columns)
    return output_df


def save_multiple_predictions(input_data, sequences, scores, cfg):
    output_list = [
        [input_data.loc[i // cfg.num_return_sequences, cfg.input_column]]
        + sequences[i : i + cfg.num_return_sequences]
        + scores[i : i + cfg.num_return_sequences]
        for i in range(0, len(sequences), cfg.num_return_sequences)
    ]
    columns = (
        ["input"]
        + [f"{i}th" for i in range(cfg.num_return_sequences)]
        + ([f"{i}th score" for i in range(cfg.num_return_sequences)] if scores else [])
    )
    output_df = pd.DataFrame(output_list, columns=columns)
    return output_df


if __name__ == "__main__":
    CFG = parse_args()
    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

    seed_everything(seed=CFG.seed)

    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path, return_tensors="pt")
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name_or_path).to(CFG.device)
    model.eval()


    if "csv" not in CFG.input_data:
        input_compound = CFG.input_data
        output = predict_single_input(input_compound, model, CFG)
        sequences, scores = decode_output(output, CFG)
        output_df = save_single_prediction(input_compound, sequences, scores, CFG)
    else:
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

        all_sequences, all_scores = [], []
        for inputs in dataloader:
            inputs = {k: v[0].to(CFG.device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    min_length=CFG.output_min_length,
                    max_length=CFG.output_max_length,
                    num_beams=CFG.num_beams,
                    num_return_sequences=CFG.num_return_sequences,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            sequences, scores = decode_output(output, CFG)
            all_sequences.extend(sequences)
            if scores:
                all_scores.extend(scores)
            del output
            torch.cuda.empty_cache()
            gc.collect()

        output_df = save_multiple_predictions(input_data, all_sequences, all_scores, CFG)

    output_df.to_csv(os.path.join(CFG.output_dir, "output.csv"), index=False)
