import os
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import argparse
from torch.utils.data import DataLoader
import sys
import numpy as np

sys.path.append("../")
from utils import seed_everything, filter_out
from generation_utils import ReactionT5Dataset
from train import preprocess_df

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the input data.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=False,
        help="Path to the test data. If provided, the duplicates will be removed from the input data.",
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
        default="sagawa/ReactionT5v2-retrosynthesis",
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
    outputs_mean = []
    model.eval()
    model.to(device)
    for inputs in dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate_embedding(inputs)
        outputs_mean.append(output.detach().cpu().numpy())

    return outputs_mean


class CustomModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def generate_embedding(self, inputs):
        encoder_outputs = self.encoder(**inputs)
        encoder_hidden_states = encoder_outputs[0]
        decoder_outputs = self.decoder(
            input_ids=torch.full(
                (inputs["input_ids"].size(0), 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=inputs["input_ids"].device,
            ),
            encoder_hidden_states=encoder_hidden_states,
        )
        encoder_hidden_states = torch.mean(encoder_hidden_states, dim=1)
        decoder_hidden_states = torch.squeeze(decoder_outputs[0], 1)

        return torch.cat((encoder_hidden_states, decoder_hidden_states), dim=1)


if __name__ == "__main__":
    CFG = parse_args()
    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

    seed_everything(seed=CFG.seed)

    CFG.tokenizer = AutoTokenizer.from_pretrained(
        CFG.model_name_or_path, return_tensors="pt"
    )
    model = CustomModel.from_pretrained(CFG.model_name_or_path).to(CFG.device)
    model.eval()

    input_data = filter_out(pd.read_csv(CFG.input_data), ["REACTANT", "PRODUCT"])
    input_data = preprocess_df(input_data, drop_duplicates=False)
    if CFG.test_data:
        test_data = filter_out(pd.read_csv(CFG.test_data), ["REACTANT", "PRODUCT"])
        test_data = preprocess_df(test_data, drop_duplicates=False)
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
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    outputs = create_embedding(dataloader, model, CFG.device)

    outputs = np.concatenate(outputs, axis=0)

    np.save(os.path.join(CFG.output_dir, "embedding_mean_v2.npy"), outputs)
