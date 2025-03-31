import argparse
import glob
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
sys.path.append("../")
from finetune import download_pretrained_model
from generation_utils import ReactionT5Dataset
from models import ReactionT5Yield
from train import inference_fn, preprocess_df
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
        help="Data as a string or CSV file that contains an 'input' column. The format of the string or contents of the column are like 'REACTANT:{reactants of the reaction}PRODUCT:{products of the reaction}'. If there are multiple reactants, concatenate them with '.'.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name or path of the finetuned model for prediction. Can be a local model or one from Hugging Face.",
    )
    parser.add_argument(
        "--download_pretrained_model",
        action="store_true",
        help="Download finetuned model from hugging face hub and use it for prediction.",
    )
    parser.add_argument("--debug", action="store_true", help="Use debug mode.")
    parser.add_argument(
        "--input_max_length",
        type=int,
        default=300,
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

    if CFG.model_name_or_path is None:
        CFG.download_pretrained_model = True

    if CFG.download_pretrained_model:
        download_pretrained_model()
        CFG.model_name_or_path = "."

    CFG.tokenizer = AutoTokenizer.from_pretrained(
        CFG.model_name_or_path, return_tensors="pt"
    )

    model = ReactionT5Yield(
        CFG,
        config_path=os.path.join(CFG.model_name_or_path, "config.pth"),
        pretrained=False,
    )
    pth_files = glob.glob(os.path.join(CFG.model_name_or_path, "*.pth"))
    for pth_file in pth_files:
        state = torch.load(
            pth_file,
            map_location=torch.device("cpu"),
        )
        try:
            model.load_state_dict(state)
            break
        except:
            pass

    if CFG.input_data.endswith(".csv"):
        test_ds = pd.read_csv(CFG.input_data)
        test_ds = preprocess_df(test_ds, CFG, drop_duplicates=False)
    else:
        test_ds = pd.DataFrame.from_dict({"input": [CFG.input_data]}, orient="index").T

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

    test_ds["prediction"] = prediction * 100
    test_ds["prediction"] = test_ds["prediction"].clip(0, 100)
    test_ds.to_csv(
        os.path.join(CFG.output_dir, "yield_prediction_output.csv"), index=False
    )
