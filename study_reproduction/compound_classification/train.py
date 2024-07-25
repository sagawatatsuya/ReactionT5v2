import os
import gc
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, T5EncoderModel, get_linear_schedule_with_warmup, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import datasets
from datasets import load_dataset, load_metric
import sentencepiece
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
import pickle
import time
import math
from sklearn.preprocessing import MinMaxScaler
from datasets.utils.logging import disable_progress_bar
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
disable_progress_bar()
import sys
sys.path.append('../../')
from utils import seed_everything, canonicalize, space_clean, get_logger, AverageMeter, asMinutes, timeSince, get_optimizer_params
from models import ClassificationT5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../data/data_for_classification_binary.csv", required=False)
    parser.add_argument("--pretrained_model_name_or_path", default="sagawa/CompoundT5", type=str, required=False)
    parser.add_argument("--model_name_or_path", type=str, required=False)
    parser.add_argument("--debug", action='store_true', default=False, required=False)
    parser.add_argument("--epochs", type=int, default=100, required=False)
    parser.add_argument("--lr", type=float, default=2e-5, required=False)
    parser.add_argument("--batch_size", type=int, default=25, required=False)
    parser.add_argument("--max_len", type=int, default=128, required=False)
    parser.add_argument("--num_workers", type=int, default=4, required=False)
    parser.add_argument("--fc_dropout", type=float, default=0.1, required=False)
    parser.add_argument("--model", type=str, default='t5', required=False)
    parser.add_argument("--eps", type=float, default=1e-6, required=False)
    parser.add_argument("--max_grad_norm", type=int, default=1000, required=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--num_warmup_steps", type=int, default=0, required=False)
    parser.add_argument("--batch_scheduler", action='store_true', default=False, required=False)
    parser.add_argument("--print_freq", type=int, default=100, required=False)
    parser.add_argument("--use_apex", action='store_true', default=False, required=False)
    parser.add_argument("--output_dir", type=str, default='./', required=False)
    parser.add_argument("--weight_decay", type=float, default=0.01, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)

    return parser.parse_args()

CFG = parse_args()
CFG.batch_scheduler = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG.device = device

OUTPUT_DIR = CFG.output_dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

seed_everything(seed=CFG.seed)  
    

df = pd.read_csv(CFG.data_path)
df['target'] *= -1
df['target'] += 1


train_ds = df[df['fold'] != 0].drop(['fold'], axis=1)
valid_ds = df[df['fold'] == 0].drop(['fold'], axis=1)

train_ds.to_csv('../../data/classification-input-train.csv', index=False)
valid_ds.to_csv('../../data/classification-input-valid.csv', index=False)

train_ds = pd.concat([train_ds, train_ds[train_ds['target'] == 1].sample(n=len(train_ds[train_ds['target'] == 1])*100, replace=True)])

if CFG.debug:
    train_ds = train_ds.sample(frac=0.2).reset_index(drop=True)
    valid_ds = valid_ds.sample(frac=0.2).reset_index(drop=True)

LOGGER = get_logger(OUTPUT_DIR+'train')

#load tokenizer
try: # load pretrained tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), return_tensors='pt')
except: # load pretrained tokenizer from huggingface model hub
    tokenizer = AutoTokenizer.from_pretrained(CFG.pretrained_model_name_or_path, return_tensors='pt')
tokenizer.add_tokens(['>', '<', 'P','Pd', 'Na', 'K', 'Al', 'Cu', 'Si', 'Zn', 'Mn', 'Li', 'Mg', 'Fe', 'Ba', 'Pt', 'Ag', 'Yb', '6', 'e'])
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False, truncation=True, return_attention_mask=True)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df['smiles'].values
        self.labels = df['target'].values
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.inputs[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        
        return inputs, label


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.use_apex):
            y_preds = model(inputs)
        loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss/CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]), flush=True)
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    start = end = time.time()
    label_list = []
    pred_list = []
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.type(torch.LongTensor)
        with torch.no_grad():
            y_preds = model(inputs)
        label_list += labels.tolist()
        pred_list += y_preds.tolist()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):

            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'accuracy: {loss:.4f} '
                  'f1 score: {f1score}'
                  .format(step, len(valid_loader),
                          loss=accuracy_score(label_list, torch.argmax(torch.Tensor(pred_list), dim=1)),
                          f1score=f1_score(label_list, torch.argmax(torch.Tensor(pred_list), dim=1)),
                          remain=timeSince(start, float(step+1)/len(valid_loader))), flush=True)
    return f1_score(label_list, torch.argmax(torch.Tensor(pred_list), dim=1))
    
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


def train_loop(train_ds, valid_ds):
    
    train_dataset = TrainDataset(CFG, train_ds)
    valid_dataset = TrainDataset(CFG, valid_ds)
    valid_labels = valid_ds['target'].values
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    model = ClassificationT5(CFG, config_path=None, pretrained=True, phase='train')
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    
    optimizer_parameters = get_optimizer_params(model, encoder_lr=CFG.lr, decoder_lr=CFG.lr, weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.lr, eps=CFG.eps, betas=(0.9, 0.999))
    
    num_train_steps = int(len(train_ds)/CFG.batch_size*CFG.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps)
    
    criterion = nn.CrossEntropyLoss()
    best_score = -float('inf')
    
    for epoch in range(CFG.epochs):
        start_time = time.time()

        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        f1_score = valid_fn(valid_loader, model, criterion, device)
        
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  val_f1_score {f1_score:.4f}  time: {elapsed:.0f}s')
    
        if best_score < f1_score:
            best_score = f1_score
            LOGGER.info(f'Epoch {epoch+1} - Save highest f1_score: {best_score:.4f} Model')
            torch.save(model.state_dict(), OUTPUT_DIR+f"{CFG.pretrained_model_name_or_path.split('/')[-1]}_best.pth")
    
    torch.cuda.empty_cache()
    gc.collect()

            
if __name__ == '__main__':
    train_loop(train_ds, valid_ds)
        
 