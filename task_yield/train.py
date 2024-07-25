import os
import gc
import warnings
import time
import sys
import glob

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from datasets.utils.logging import disable_progress_bar
from sklearn.metrics import mean_squared_error, r2_score

# Append the utils module path
sys.path.append("../")
from utils import (
    seed_everything,
    canonicalize,
    get_logger,
    AverageMeter,
    timeSince,
    get_optimizer_params,
    space_clean
)
from models import ReactionT5Yield

# Suppress warnings and logging
warnings.filterwarnings("ignore")
disable_progress_bar()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training script for ReactionT5Yield model."
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to training data CSV file.",
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        required=True,
        help="Path to validation data CSV file.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to testing data CSV file.",
    )
    parser.add_argument(
        "--CN_test_data_path",
        type=str,
        required=True,
        help="Path to CN testing data CSV file.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sagawa/CompoundT5",
        help="Pretrained model name or path.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The model's name or path used for fine-tuning.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience."
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size.")
    parser.add_argument(
        "--max_len", type=int, default=400, help="Maximum input token length."
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
        "--eps", type=float, default=1e-6, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=int,
        default=1000,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of warmup steps."
    )
    parser.add_argument(
        "--batch_scheduler", action="store_true", help="Use batch scheduler."
    )
    parser.add_argument(
        "--print_freq", type=int, default=100, help="Logging frequency."
    )
    parser.add_argument(
        "--use_apex", action="store_true", help="Use apex for mixed precision training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint file for resuming training.",
    )

    return parser.parse_args()


def preprocess_df(df, cfg):
    """
    Preprocess the input DataFrame for training.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cfg (argparse.Namespace): Configuration object.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = df[~df["YIELD"].isna()].reset_index(drop=True)
    df["YIELD"] = df["YIELD"].apply(lambda x: 100 if x > 100 else x) / 100
    df = df[~(df["REACTANT"].isna() | df["PRODUCT"].isna())]

    for col in [
        "CATALYST",
        "REACTANT",
        "REAGENT",
        "SOLVENT",
        "INTERNAL_STANDARD",
        "NoData",
        "PRODUCT",
    ]:
        df[col] = df[col].fillna(" ")

    df["REAGENT"] = df["CATALYST"] + "." + df["REAGENT"]
    df["REAGENT"] = df["REAGENT"].apply(lambda x: space_clean(x))
    df["REAGENT"] = df["REAGENT"].apply(lambda x: canonicalize(x) if x != " " else " ")
    df["REACTANT"] = df["REACTANT"].apply(lambda x: ".".join(sorted(x.split("."))))
    df["REAGENT"] = df["REAGENT"].apply(lambda x: ".".join(sorted(x.split("."))))
    df["PRODUCT"] = df["PRODUCT"].apply(lambda x: ".".join(sorted(x.split("."))))

    df["input"] = (
        "REACTANT:"
        + df["REACTANT"]
        + "REAGENT:"
        + df["REAGENT"]
        + "PRODUCT:"
        + df["PRODUCT"]
    )
    df = df.loc[df[["input", "YIELD"]].drop_duplicates().index].reset_index(drop=True)

    if cfg.debug:
        df = df.head(1000)

    return df


def preprocess_CN(df):
    """
    Preprocess the CN test DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df["REACTANT"] = df["REACTANT"].apply(lambda x: ".".join(sorted(x.split("."))))
    df["REAGENT"] = df["REAGENT"].apply(lambda x: ".".join(sorted(x.split("."))))
    df["PRODUCT"] = df["PRODUCT"].apply(lambda x: ".".join(sorted(x.split("."))))
    df["input"] = (
        "REACTANT:"
        + df["REACTANT"]
        + "REAGENT:"
        + df["REAGENT"]
        + "PRODUCT:"
        + df["PRODUCT"]
    )
    df["pair"] = df["input"]
    return df


def prepare_input(cfg, text):
    """
    Prepare input tensors for the model.

    Args:
        cfg (argparse.Namespace): Configuration object.
        text (str): Input text.

    Returns:
        dict: Tokenized input tensors.
    """
    inputs = cfg.tokenizer(
        text,
        add_special_tokens=True,
        max_length=cfg.max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    return {k: torch.tensor(v, dtype=torch.long) for k, v in inputs.items()}


class TrainDataset(Dataset):
    """
    Dataset class for training.
    """

    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df["input"].values
        self.labels = df["YIELD"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.inputs[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Save model checkpoint.

    Args:
        state (dict): Checkpoint state.
        filename (str): Filename to save the checkpoint.
    """
    torch.save(state, filename)


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, cfg):
    """
    Training function for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): Model to be trained.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        epoch (int): Current epoch.
        scheduler (Scheduler): Learning rate scheduler.
        cfg (argparse.Namespace): Configuration object.

    Returns:
        float: Average training loss.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_apex)
    losses = AverageMeter()
    start = time.time()

    for step, (inputs, labels) in enumerate(train_loader):
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
        labels = labels.to(cfg.device)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=cfg.use_apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        if cfg.gradient_accumulation_steps > 1:
            loss /= cfg.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if cfg.batch_scheduler:
                scheduler.step()

        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader))} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Grad: {grad_norm:.4f} "
                f"LR: {scheduler.get_lr()[0]:.8f}"
            )

    return losses.avg


def valid_fn(valid_loader, model, cfg):
    """
    Validation function.

    Args:
        valid_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): Model to be validated.
        cfg (argparse.Namespace): Configuration object.

    Returns:
        tuple: Validation loss and R^2 score.
    """
    model.eval()
    start = time.time()
    label_list = []
    pred_list = []

    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
        with torch.no_grad():
            y_preds = model(inputs)
        label_list.extend(labels.tolist())
        pred_list.extend(y_preds.tolist())

        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Elapsed {timeSince(start, float(step+1)/len(valid_loader))} "
                f"RMSE Loss: {mean_squared_error(label_list, pred_list, squared=False):.4f} "
                f"R^2 Score: {r2_score(label_list, pred_list):.4f}"
            )

    return mean_squared_error(label_list, pred_list), r2_score(label_list, pred_list)


def inference_fn(test_loader, model, cfg):
    """
    Inference function.

    Args:
        test_loader (DataLoader): DataLoader for test data.
        model (nn.Module): Model for inference.
        cfg (argparse.Namespace): Configuration object.

    Returns:
        np.ndarray: Predictions.
    """
    model.eval()
    model.to(cfg.device)
    preds = []

    for inputs in tqdm(test_loader, total=len(test_loader)):
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to("cpu").numpy())

    return np.concatenate(preds)


def train_loop(train_ds, valid_ds, cfg):
    """
    Training loop.

    Args:
        train_ds (pd.DataFrame): Training data.
        valid_ds (pd.DataFrame): Validation data.
    """
    train_dataset = TrainDataset(cfg, train_ds)
    valid_dataset = TrainDataset(cfg, valid_ds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if not cfg.model_name_or_path:
        model = ReactionT5Yield(cfg, config_path=None, pretrained=True)
        torch.save(model.config, os.path.join(cfg.output_dir, "config.pth"))
    else:
        model = ReactionT5Yield(
            cfg, config_path=os.path.join(cfg.model_name_or_path, "config.pth"), pretrained=False
        )
        pth_files = glob.glob(os.path.join(cfg.model_name_or_path, "*.pth"))
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
    model.to(cfg.device)

    optimizer_parameters = get_optimizer_params(
        model, encoder_lr=cfg.lr, decoder_lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    optimizer = AdamW(optimizer_parameters, lr=cfg.lr, eps=cfg.eps, betas=(0.9, 0.999))

    num_train_steps = int(len(train_ds) / cfg.batch_size * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    criterion = nn.MSELoss(reduction="mean")
    best_loss = float("inf")
    start_epoch = 0
    es_count = 0

    if cfg.checkpoint:
        checkpoint = torch.load(cfg.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_loss = checkpoint["loss"]
        start_epoch = checkpoint["epoch"] + 1
        es_count = checkpoint["es_count"]
        del checkpoint

    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()

        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, scheduler, cfg
        )
        val_loss, val_r2_score = valid_fn(valid_loader, model, cfg)

        elapsed = time.time() - start_time

        cfg.logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  val_rmse_loss: {val_loss:.4f}  val_r2_score: {val_r2_score:.4f}  time: {elapsed:.0f}s"
        )

        if val_loss < best_loss:
            es_count = 0
            best_loss = val_loss
            cfg.logger.info(f"Epoch {epoch+1} - Save Lowest Loss: {best_loss:.4f} Model")
            torch.save(
                model.state_dict(),
                os.path.join(
                    cfg.output_dir,
                    f"{cfg.pretrained_model_name_or_path.split('/')[-1]}_best.pth",
                ),
            )
        else:
            es_count += 1
            if es_count >= cfg.patience:
                print("Early stopping")
                break

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": best_loss,
                "es_count": es_count,
            },
            filename=os.path.join(cfg.output_dir, "checkpoint.pth.tar"),
        )

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    CFG = parse_args()
    CFG.batch_scheduler = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG.device = device
    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)
    seed_everything(seed=CFG.seed)

    train = pd.read_csv(CFG.train_data_path)
    valid = pd.read_csv(CFG.valid_data_path)
    CN_test = pd.read_csv(CFG.CN_test_data_path)

    train = preprocess_df(train, CFG)
    valid = preprocess_df(valid, CFG)
    train_copy = preprocess_CN(train.copy())
    CN_test = preprocess_CN(CN_test)

    print(len(train))
    train = train[~train_copy["pair"].isin(CN_test["pair"])].reset_index(drop=True)
    print(len(train))

    train["pair"] = train["input"] + " - " + train["YIELD"].astype(str)
    valid["pair"] = valid["input"] + " - " + valid["YIELD"].astype(str)
    valid = valid[~valid["pair"].isin(train["pair"])].reset_index(drop=True)

    train.to_csv("train.csv", index=False)
    valid.to_csv("valid.csv", index=False)

    if CFG.test_data_path:
        test = pd.read_csv(CFG.test_data_path)
        test = preprocess_df(test, CFG)
        test["pair"] = test["input"] + " - " + test["YIELD"].astype(str)
        test = test[~test["pair"].isin(train["pair"])].reset_index(drop=True)
        test.to_csv("test.csv", index=False)

    LOGGER = get_logger(os.path.join(CFG.output_dir, "train"))
    CFG.logger = LOGGER

    # load tokenizer
    try:  # load pretrained tokenizer from local directory
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.abspath(CFG.pretrained_model_name_or_path), return_tensors="pt"
        )
    except:  # load pretrained tokenizer from huggingface model hub
        tokenizer = AutoTokenizer.from_pretrained(
            CFG.pretrained_model_name_or_path, return_tensors="pt"
        )
    tokenizer.add_tokens(
        [
            ".",
            "6",
            "7",
            "8",
            "<",
            ">",
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "Ba",
            "Bi",
            "Ca",
            "Cl",
            "Cu",
            "Fe",
            "Ge",
            "Hg",
            "K",
            "Li",
            "Mg",
            "Mn",
            "Mo",
            "Na",
            "Nd",
            "Ni",
            "P",
            "Pb",
            "Pd",
            "Pt",
            "Re",
            "Rh",
            "Ru",
            "Ru",
            "Sb",
            "Si",
            "Sm",
            "Ta",
            "Ti",
            "Tl",
            "W",
            "Yb",
            "Zn",
            "Zr",
            "e",
            "p",
        ]
    )

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": tokenizer.additional_special_tokens
            + ["REACTANT:", "PRODUCT:", "REAGENT:"]
        }
    )
    tokenizer.save_pretrained(os.path.join(CFG.output_dir, "tokenizer/"))
    CFG.tokenizer = tokenizer

    train_loop(train, valid, CFG)
