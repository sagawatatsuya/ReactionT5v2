import os
import random
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
import pickle
from datasets.utils.logging import disable_progress_bar
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
disable_progress_bar()
import sys
sys.path.append('../../')
from utils import seed_everything, canonicalize, space_clean, get_logger, AverageMeter, asMinutes, timeSince
from models import ClassificationT5

class CFG():
    data_path = '../../data/nodata.csv'
    pretrained_model_name_or_path='./tokenizer'
    debug = True
    model = 't5'
    epochs = 5
    lr = 2e-5
    batch_size = 64
    max_len = 128
    weight_decay = 0.01
    seed = 42
    num_workers = 4
    fc_dropout = 0.1
    eps = 1e-6
    max_grad_norm=1000
    gradient_accumulation_steps=1
    num_warmup_steps=0
    batch_scheduler=True
    print_freq=100
    use_apex=False
    output_file = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG.device = device

seed_everything(seed=CFG.seed)  


#load tokenizer
try: # load pretrained tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), return_tensors='pt')
except: # load pretrained tokenizer from huggingface model hub
    tokenizer = AutoTokenizer.from_pretrained(CFG.pretrained_model_name_or_path, return_tensors='pt')

CFG.tokenizer = tokenizer
def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False, truncation=True, return_attention_mask=True)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    
    return inputs

    
class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df['smiles'].values
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.inputs[item])
        
        return inputs

    
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


valid_ds = pd.read_csv(CFG.data_path)
CFG.model_name_or_path = 't5'
valid_dataset = TestDataset(CFG, valid_ds)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
model = ClassificationT5(CFG, config_path='./config.pth', pretrained=False, phase='test')
state = torch.load('./ZINC-t5_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(state)

model.to(device)
preds = []
for inputs in valid_loader:
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        y_preds = model(inputs)
        preds.append(y_preds.tolist())

with open('pred.pkl', 'bw') as f:
    pickle.dump(preds, f)
    
    
lis = []
for p in pred:
    lis += p
lis = np.array(lis)
pred = lis[:, 1] > 0.97
valid_ds['pred'] = pred
valid_ds.to_csv('../../data/nodata-prediction.csv', index=False)