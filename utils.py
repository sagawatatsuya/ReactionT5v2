import os
import random
import numpy as np
import torch
from rdkit import Chem
import math
import time

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def space_clean(row):
    row = row.replace('. ', '').replace(' .', '').replace('  ', ' ')
    return row

    
def canonicalize(smiles):
    try:
        new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    except:
        new_smiles = None
    return new_smiles


def canonicalize_str(smiles):
    """Try to canonicalize the molecule, return empty string if fails."""
    try:
        return canonicalize(smiles)
    except:
        if "%" in smiles:
            return smiles
        else:
            return ""


def uncanonicalize(smiles):
    try:
        new_smiles = []
        for smiles_i in smiles.split('.'):
            mol = Chem.MolFromSmiles(smiles_i)
            atom_indices = list(range(mol.GetNumAtoms()))
            random.shuffle(atom_indices)
            new_smiles_i = Chem.MolToSmiles(mol, rootedAtAtom=atom_indices[0], canonical=False)
            new_smiles.append(new_smiles_i)
        smiles = '.'.join(new_smiles)
    except:
        smiles = None
    return smiles
    

def remove_atom_mapping(smi):
    mol = Chem.MolFromSmiles(smi)
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    smi = Chem.MolToSmiles(mol, canonical=True)
    return canonicalize(smi)


def get_logger(filename='train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
        

def asMinutes(s):
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s/(percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)], 'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 'model' not in n], 'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.to('cpu')
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set) or isinstance(obj, torch.Tensor):
        return [to_cpu(v) for v in obj]
    else:
        return obj
    

def get_accuracy_score(eval_preds, cfg):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = cfg.tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, cfg.tokenizer.pad_token_id)
    decoded_labels = cfg.tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [canonicalize_str(pred.strip().replace(' ', '')) for pred in decoded_preds]
    decoded_labels = [[canonicalize_str(label.strip().replace(' ', ''))] for label in decoded_labels]

    score = 0
    for i in range(len(decoded_preds)):
        if decoded_preds[i] == decoded_labels[i][0]:
            score += 1
    score /= len(decoded_preds)
    return {'accuracy': score}


def get_accuracy_score_multitask(eval_preds, cfg):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    special_tokens = cfg.tokenizer.special_tokens_map
    special_tokens = [special_tokens['eos_token'], special_tokens['pad_token'], special_tokens['unk_token']] + list(set(special_tokens['additional_special_tokens']) - set(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']))

    decoded_preds = cfg.tokenizer.batch_decode(preds, skip_special_tokens=False)
    for special_token in special_tokens:
        decoded_preds = [pred.replace(special_token, '') for pred in decoded_preds]

    labels = np.where(labels != -100, labels, cfg.tokenizer.pad_token_id)
    decoded_labels = cfg.tokenizer.batch_decode(labels, skip_special_tokens=False)
    for special_token in special_tokens:
        decoded_labels = [pred.replace(special_token, '') for pred in decoded_labels]

    decoded_preds = [canonicalize_str(pred.strip().replace(' ', '')) for pred in decoded_preds]
    decoded_labels = [[canonicalize_str(label.strip().replace(' ', ''))] for label in decoded_labels]

    score = 0
    for i in range(len(decoded_preds)):
        if decoded_preds[i] == decoded_labels[i][0]:
            score += 1
    score /= len(decoded_preds)
    return {'accuracy': score}


def preprocess_dataset(examples, cfg):
    inputs = examples['input']
    targets = examples[cfg.target_column]
    model_inputs = cfg.tokenizer(inputs, max_length=cfg.input_max_length, truncation=True)
    labels = cfg.tokenizer(targets, max_length=cfg.target_max_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs