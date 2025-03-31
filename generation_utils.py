import pandas as pd
import torch
from torch.utils.data import Dataset


def prepare_input(cfg, text):
    inputs = cfg.tokenizer(
        text,
        add_special_tokens=True,
        max_length=cfg.input_max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    return {k: torch.tensor(v, dtype=torch.long) for k, v in inputs.items()}


class ReactionT5Dataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df["input"].values

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
        [input_data.loc[i // cfg.num_return_sequences, "input"]]
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
