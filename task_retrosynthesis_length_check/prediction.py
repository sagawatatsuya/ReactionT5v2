import os
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
from torch.utils.data import DataLoader
import sys
import gc

sys.path.append("../")
from utils import seed_everything
from generation_utils import ReactionT5Dataset, predict_single_input, decode_output, save_single_prediction, save_multiple_predictions
from train import preprocess_df

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
        default="sagawa/ReactionT5v2-retrosynthesis",
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
    parser.add_argument(
        "--diff", type=int, default=0, help="Diff from L"
    )
    return parser.parse_args()


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
        input_data = preprocess_df(input_data, drop_duplicates=False)
        dataset = ReactionT5Dataset(CFG, input_data)
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
            inputs = {k: v.to(CFG.device) for k, v in inputs.items()}
            L = inputs["input_ids"][0].tolist().index(CFG.tokenizer.eos_token_id)
            with torch.no_grad():
                # output = model.generate(
                #     **inputs,
                #     min_length=max(0, L - CFG.diff),
                #     max_length=L + CFG.diff,
                #     num_beams=CFG.num_beams,
                #     num_return_sequences=CFG.num_return_sequences,
                #     return_dict_in_generate=True,
                #     output_scores=True,
                # )
                output = model.generate(
                    **inputs,
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
