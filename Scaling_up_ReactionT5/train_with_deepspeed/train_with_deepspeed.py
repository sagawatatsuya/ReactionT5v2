#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""
import gc
import glob
import json
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partialmethod
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, disable_caching, load_dataset, load_from_disk
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
from transformers import (Adafactor, AutoTokenizer, BatchEncoding,
                          HfArgumentParser, PreTrainedTokenizerBase, T5Config,
                          T5ForConditionalGeneration, set_seed, Trainer, TrainingArguments, EarlyStoppingCallback)
from torch.optim.lr_scheduler import LRScheduler

sys.path.append('/home/acf15718oa/train_with_deepspeed')
from utils import get_logger
# import deepspeed
# from deepspeed.accelerator import get_accelerator
# import torch.distributed as dist
# import torch.multiprocessing as mp


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
# disable_caching()
# os.environ['TRANSFORMERS_CACHE'] = '/groups/gca50095/sagawa/cache/'
# os.environ['HF_DATASETS_CACHE'] = "/groups/gca50095/sagawa/cache/"


@dataclass
class TrainingArguments_:
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    output_dir: str = field(
        default="./output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    checkpoint: str = field(default=None, metadata={"help": "Path to checkpoint to resume training from."})
    save_total_limit: int = field(default=None, metadata={"help": "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir. Default is unlimited checkpoints."})
    local_rank: int = field(default=-1, metadata={"help": "Local rank of the process on the node."})
    deepspeed: str = field(default=None, metadata={"help": "DeepSpeed json config file."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing to save memory at the expense of slower backward pass."})


    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_files_dir: Optional[str] = field(default=None, metadata={"help": "The directory where input training data file is stored."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the data processing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length
    

def shift_tokens_right(input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            noise_density: float,
            mean_noise_span_length: float,
            input_length: int,
            target_length: int,
            pad_token_id: int,
            decoder_start_token_id: int,
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id


    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # for k, v in examples[0].items():
        #     print(k)
        #     for i in range(len(examples)):
        #         print(np.array(examples[i][k]).shape)

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )
        # convert all item from numpy array to pytorch tensor
        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


# class InverseSqrtScheduler(LRScheduler):
#     def __init__(self, optimizer, warmup_steps=0, last_epoch=-1, warmup_init_lr=0, verbose=False):
#         self.optimizer = optimizer
#         self.warmup_steps = warmup_steps
#         self.warmup_init_lr = warmup_init_lr
#         warmup_end_lr = self.optimizer.param_groups[0]['lr']
#         if self.warmup_steps > 0:
#             self.lr_step = (warmup_end_lr - warmup_init_lr) / self.warmup_steps
#         super(InverseSqrtScheduler, self).__init__(optimizer, last_epoch, verbose)

#     def get_lr(self):
#         if self._step_count < self.warmup_steps:
#             return [self.warmup_init_lr + self.lr_step * self._step_count for base_lr in self.base_lrs]
#         else:
#             return [base_lr / np.sqrt(self._step_count) for base_lr in self.base_lrs]

class InverseSqrtScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps=0, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        super(InverseSqrtScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return self.base_lrs
        else:
            return [base_lr / np.sqrt(self._step_count) for base_lr in self.base_lrs]


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

        self.lr_scheduler = InverseSqrtScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.args.warmup_steps,
            last_epoch=-1,
        )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments_))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    try:
        os.mkdir(training_args.output_dir)
    except:
        print('output_dir already exists')

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # Load pretrained model and tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.config_name:
        config = T5Config.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            vocab_size=len(tokenizer),
        )
    elif model_args.model_name_or_path:
        config = T5Config.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = T5ForConditionalGeneration(config=config)

    # Get the datasets
    data_files = glob.glob(data_args.train_files_dir+'*_drop_last_row.parquet')

    # if H24, H21, H4, or H5 is in the file name, it is a validation file
    train_data_files = [f for f in data_files if 'H24' not in f and 'H21' not in f and 'H4' not in f and 'H5' not in f]
    val_data_files = [f for f in data_files if 'H21' in f]

    # train_data_files = ['/data/sagawa/preprocessed_ZINC22/H17_drop_last_row.parquet', '/data/sagawa/preprocessed_ZINC22/H06_drop_last_row.parquet', '/data/sagawa/preprocessed_ZINC22/H07_drop_last_row.parquet']
    # val_data_files = ['/data/sagawa/preprocessed_ZINC22/H17_drop_last_row.parquet', '/data/sagawa/preprocessed_ZINC22/H06_drop_last_row.parquet', '/data/sagawa/preprocessed_ZINC22/H07_drop_last_row.parquet']


    random.seed(training_args.seed)
    random.shuffle(data_files)

    datasets = load_dataset(
        'parquet',
        data_files={'train': train_data_files, 'validation': val_data_files},
        cache_dir=None,
        streaming=True
    )


    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
    # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
    # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
    )


    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    # fill the default training arguments
    args = TrainingArguments(
        output_dir=training_args.output_dir, #"/home/acf15718oa/T5configs/small",
        overwrite_output_dir=training_args.overwrite_output_dir,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        evaluation_strategy="steps",
        save_total_limit=int(training_args.save_total_limit),
        per_device_train_batch_size=int(training_args.per_device_train_batch_size),
        per_device_eval_batch_size=int(training_args.per_device_eval_batch_size),
        gradient_accumulation_steps=1,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        num_train_epochs=int(training_args.num_train_epochs),
        warmup_steps=training_args.warmup_steps,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        seed=training_args.seed,
        data_seed=training_args.seed,
        fp16=True,
        local_rank=training_args.local_rank,
        dataloader_num_workers=data_args.num_workers,
        deepspeed=training_args.deepspeed,
        gradient_checkpointing=training_args.gradient_checkpointing,
        auto_find_batch_size=False,
        max_steps=int((3548266479 - 21316886)*training_args.num_train_epochs/(int(training_args.per_device_train_batch_size*training_args.local_rank))),
        load_best_model_at_end=True,
    )


    trainer = MyTrainer(
        model,
        args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print('training started')

    trainer.train(resume_from_checkpoint=training_args.checkpoint)

if __name__ == "__main__":
    main()