import os
import gc
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import pickle
import time
import math
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
import sys
sys.path.append('../../')
from utils import seed_everything
seed_everything(seed=42)  


df = pd.read_csv('../../data/nodata-prediction.csv')
dic = {}
for idx, row in df.iterrows():
    dic[row['smiles']] = row['pred']
    
df_all = pd.read_csv('../../data/all_ord_reaction_uniq_with_attr_v3.csv')
df_all = df_all[~df_all['PRODUCT'].isna()]
df_all = df_all[df_all['REACTANT'].isna()].reset_index(drop=True)

dfs = []
name2num = {'REAGENT': 0,
 'REACTANT': 1}
names = ['REAGENT', 'REACTANT']
for idx, row in df_all.iterrows():
    lis = []
    if type(row['NoData']) != str:
        lis = row[names].values.tolist()
    else:
        for name in names:
            if type(row[name]) == str:
                lis.append([row[name]])
            else:
                lis.append([])
        for smile in row['NoData'].split('.'):
            num = dic[smile]
            lis[num].append(smile)
        lis = ['.'.join(smiles) for smiles in lis]
    dfs.append(lis)

df_after = pd.DataFrame(np.array(dfs).reshape(-1, 2), columns=names)

df_after[['PRODUCT', 'YIELD', 'TEMP', 'NoData']] = df_all[['PRODUCT', 'YIELD', 'TEMP', 'NoData']]

df_after.to_csv('../../data/reconstructed-nodata.csv', index=False)

print('finish')