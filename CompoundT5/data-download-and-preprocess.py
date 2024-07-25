import subprocess
from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import canonicalize, remove_atom_mapping

subprocess.run("mkdir data", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1ZPsoUYb4HcxFzK_ac9rb_pQj7oO3Gagh'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1XwkxxHiaWFbSNhGyxnv6hAliutIMNrIp'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1yIwUH_OhER9nuMo9HjBhBmyc6zvmrSPA'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1skFRirstIUijhieshvJEScBD2aB3H1YU'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1snkL0YPQOEVr-7JuwuR-HlCECV_Zwp7l'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1Qbsl8_CmdIK_iNNY8F6wATVnDQNSW9Tc'", shell=True)
subprocess.run("mv *.smi data", shell=True)
subprocess.run("mv *.txt data", shell=True)
subprocess.run("mv *.tsv data", shell=True)


lst = set([])
for i in range(4):
    suppl = Chem.SmilesMolSupplier(f'../data/16_p{i}.smi')
    # some molecules cannot be read by rdkit, so we remove them here
    for mol in suppl:
        try:
            sm = Chem.MolToSmiles(mol)
            lst.add(sm)
        except:
            pass
df = pd.DataFrame({'smiles': list(lst)})
df.to_csv('../data/ZINC-canonicalized.csv', index=False)


tx = pd.read_csv('../data/pubchem-10m.txt', names=['smiles'])
lst = set([])
for smiles in tx['smiles']:
    try:
        lst.add(canonicalize(smiles))
    except:
        pass
df = pd.DataFrame({'smiles': list(lst)})
df.to_csv('../data/pubchem-10m-canonicalized.csv', index=False)

    
df = pd.read_csv('../data/all_ord_reaction_uniq_with_attr_v1.tsv', sep='\t', names=['id', 'input', 'product', 'condition'])


def data_split(row):
    dic = {'CATALYST': [], 'REACTANT': [], 'REAGENT': [], 'SOLVENT': [], 'INTERNAL_STANDARD': [], 'NoData': []}
    inp_cat = ['CATALYST', 'REACTANT', 'REAGENT', 'SOLVENT', 'INTERNAL_STANDARD', 'NoData']
    inp = row['input']
    if type(inp) == str:
        for item in inp.split('.'):
            for cat in inp_cat:
                if cat in item:
                    dic[cat].append(item[item.find(':')+1:])
                    break
    for k, v in dic.items():
        dic[k] = '.'.join(dic[k])

    pro = row['product']
    if type(pro) == str:
        pro = pro.replace('.PRODUCT', 'PRODUCT')
        pro_lis = []
        for item in pro.split('PRODUCT:'):
            if item != '':
                pro_lis.append(item)
        dic['PRODUCT'] = '.'.join(pro_lis)
    else:
        dic['PRODUCT'] = None
    
    con = row['condition']
    if type(con) == str:
        if 'YIELD' in con and 'TEMP' in con:
            pos = con.find('.T')
            for item, cat in zip([con[:pos], con[pos:]], ['YIELD', 'TEMP']):   
                dic[cat] = float(item[item.find(':')+1:])
        elif 'YIELD' in con:
            dic['YIELD'] = float(con[con.find(':')+1:])
            dic['TEMP'] = None
        elif 'TEMP' in con:
            dic['YIELD'] = None
            dic['TEMP'] = float(con[con.find(':')+1:])
        else:
            print(con)
    else:
        for cat in ['YIELD', 'TEMP']:
            dic[cat] = None
    return list(dic.values())


dic = {'CATALYST': [], 'REACTANT': [], 'REAGENT': [], 'SOLVENT': [], 'INTERNAL_STANDARD': [], 'NoData': [], 'PRODUCT': [],'YIELD': [], 'TEMP': []}
cat = ['CATALYST', 'REACTANT', 'REAGENT', 'SOLVENT', 'INTERNAL_STANDARD', 'NoData','PRODUCT', 'YIELD', 'TEMP']
for idx, row in df.iterrows():
    lst = data_split(row)
    for i in range(len(lst)):
        dic[cat[i]].append(lst[i]) 
cleaned_df = pd.DataFrame(dic)

cleaned_df['CATALYST'] = cleaned_df['CATALYST'].apply(lambda x: remove_atom_mapping(x) if type(x) == str else None)
cleaned_df['REACTANT'] = cleaned_df['REACTANT'].apply(lambda x: remove_atom_mapping(x) if type(x) == str else None)
cleaned_df['REAGENT'] = cleaned_df['REAGENT'].apply(lambda x: remove_atom_mapping(x) if type(x) == str else None)
cleaned_df['SOLVENT'] = cleaned_df['SOLVENT'].apply(lambda x: remove_atom_mapping(x) if type(x) == str else None)
cleaned_df['INTERNAL_STANDARD'] = cleaned_df['INTERNAL_STANDARD'].apply(lambda x: remove_atom_mapping(x) if type(x) == str else None)
cleaned_df['NoData'] = cleaned_df['NoData'].apply(lambda x: remove_atom_mapping(x) if type(x) == str else None)
cleaned_df['PRODUCT'] = cleaned_df['PRODUCT'].apply(lambda x: remove_atom_mapping(x) if type(x) == str else None)


cleaned_df.to_csv('../data/all_ord_reaction_uniq_with_attr_v3.csv', index=False)