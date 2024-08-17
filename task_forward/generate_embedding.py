import os
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, T5EncoderModel
import argparse
from torch.utils.data import DataLoader
import sys
import gc

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


def predict_single_input(input_compound):
    inp = tokenizer(input_compound, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inp,
            min_length=CFG.output_min_length,
            max_length=CFG.output_max_length,
            num_beams=CFG.num_beams,
            num_return_sequences=CFG.num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return output

def create_embedding(dataloader, model, device):
    outputs = []
    outputs_cls = []
    model.eval()
    model.to(device)
    tk0 = tqdm(dataloader, total=len(dataloader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            output = model(**inputs)
        outputs.append(output[0].detach().cpu().numpy())
        outputs_cls.append(output[0][:, 0, :].detach().cpu().numpy())
    
    return outputs, outputs_cls


if __name__ == "__main__":
    CFG = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

    seed_everything(seed=CFG.seed)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path, return_tensors="pt")
    model = T5EncoderModel.from_pretrained(CFG.model_name_or_path).to(device)
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

    all_sequences, all_scores = [], []
    for inputs in dataloader:
        inputs = {k: v[0].to(device) for k, v in inputs.items()}
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
        sequences, scores = decode_output(output)
        all_sequences.extend(sequences)
        if scores:
            all_scores.extend(scores)
        del output
        torch.cuda.empty_cache()
        gc.collect()

    output_df = save_multiple_predictions(input_data, all_sequences, all_scores)

    output_df.to_csv(os.path.join(CFG.output_dir, "output.csv"), index=False)
