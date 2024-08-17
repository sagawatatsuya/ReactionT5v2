import os
import random
import numpy as np
import torch
import math
import time
import pickle

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
    smi = Chem.MolToSmiles(mol)
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
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)], 'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 'model' not in n], 'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


# def pickle_file(path, contents):
#     with open(path, 'wb') as f:
#         pickle.dump(contents, f)


# def load_pickle(path):
#     with open(path, 'rb') as f:
#         contents = pickle.load(f)
#     return contents