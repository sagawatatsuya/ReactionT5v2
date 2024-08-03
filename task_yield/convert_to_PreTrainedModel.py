import os
import sys
import argparse
import glob

import torch
from transformers import AutoTokenizer, AutoConfig

# Append the utils module path
sys.path.append("../")
from models import ReactionT5Yield


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="ReactionT5Yield model impremented with nn.Module with transformers' PreTrainedModel"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The name of a finetuned model or path to a model which you want to convert. You can use your local models or models uploaded to hugging face.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        help="The name of the base model of the finetuned model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the prediction.",
    )
    parser.add_argument(
        "--fc_dropout",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    CFG = parse_args()

    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

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

    config = AutoConfig.from_pretrained(CFG.base_model_name_or_path)
    config.vocab_size = len(CFG.tokenizer)

    CFG.tokenizer.save_pretrained(CFG.output_dir)
    torch.save(model.state_dict(), os.path.join(CFG.output_dir, "pytorch_model.bin"))
    config.save_pretrained(CFG.output_dir)