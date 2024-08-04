# https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/t5_tokenizer_model.py

import os
import sys
import json
import argparse
from typing import Iterator, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, T5Config
import datasets
from datasets import load_dataset
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    normalizers,
    pre_tokenizers,
    trainers,
)
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing

sys.path.append("../")
from utils import seed_everything

seed_everything(seed=42)


class SentencePieceUnigramTokenizer(BaseTokenizer):
    """
    This class is a copy of `DeDLOC's tokenizer implementation <https://github.com/yandex-research/DeDLOC/blob/main/sahajbert/tokenizer/tokenizer_model.py>`__ .

    Custom SentencePiece Unigram Tokenizer with NMT, NKFC, spaces and lower-casing characters normalization
    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(
        self,
        replacement: str = "▁",
        add_prefix_space: bool = True,
        unk_token: Union[str, AddedToken] = "<unk>",
        eos_token: Union[str, AddedToken] = "</s>",
        pad_token: Union[str, AddedToken] = "<pad>",
    ):
        self.special_tokens = {
            "pad": {"id": 0, "token": pad_token},
            "eos": {"id": 1, "token": eos_token},
            "unk": {"id": 2, "token": unk_token},
        }

        self.special_tokens_list = [None] * len(self.special_tokens)
        for token_dict in self.special_tokens.values():
            self.special_tokens_list[token_dict["id"]] = token_dict["token"]

        tokenizer = Tokenizer(Unigram())

        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Nmt(),
                normalizers.NFKC(),
                normalizers.Replace(Regex(" {2,}"), " "),
                #                 normalizers.Lowercase(),
            ]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Metaspace(
                    replacement=replacement, add_prefix_space=add_prefix_space
                ),
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.Punctuation(),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space
        )

        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {self.special_tokens['eos']['token']}",
            special_tokens=[
                (self.special_tokens["eos"]["token"], self.special_tokens["eos"]["id"])
            ],
        )

        parameters = {
            "model": "SentencePieceUnigram",
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given files"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

        self.add_unk_id()

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given iterator"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

        self.add_unk_id()

    def add_unk_id(self):
        tokenizer_json = json.loads(self._tokenizer.to_str())

        tokenizer_json["model"]["unk_id"] = self.special_tokens["unk"]["id"]

        self._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


def create_normal_tokenizer(dataset, model_name):
    if isinstance(dataset, datasets.dataset_dict.DatasetDict):
        training_corpus = (
            dataset["train"][i : i + 1000]["smiles"]
            for i in range(0, len(dataset), 1000)
        )
    else:
        training_corpus = (
            dataset[i : i + 1000]["smiles"] for i in range(0, len(dataset), 1000)
        )

    if "deberta" in model_name:
        # Train tokenizer
        old_tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 1000)
    elif "t5" in model_name:
        tokenizer = SentencePieceUnigramTokenizer(
            unk_token="<unk>", eos_token="</s>", pad_token="<pad>"
        )
        tokenizer.train_from_iterator(training_corpus, 1000)

    return tokenizer


def create_character_level_tokenizer(dataset, model_name):
    df = dataset["train"].to_pandas()
    df["smiles"] = [" ".join(list(i)) for i in df["smiles"]]
    dataset = datasets.Dataset.from_pandas(df)

    tokenizer = create_normal_tokenizer(dataset, model_name)

    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_character_level_tokenizer",
        action="store_true",
        default=False,
        required=False,
    )
    return parser.parse_args()


CFG = parse_args()


# Initialize a dataset
dataset = load_dataset("csv", data_files="../data/ZINC-canonicalized.csv")

if CFG.use_character_level_tokenizer:
    tokenizer = create_character_level_tokenizer(dataset, "t5")
else:
    tokenizer = create_normal_tokenizer(dataset, "t5")
# Save files to disk
tokenizer.save("./compound_pretraining/CompoundT5/CompoundT5-config/tokenizer.json")

config = T5Config.from_pretrained(
    "google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size()
)
config.save_pretrained("./compound_pretraining/CompoundT5/CompoundT5-config/")
