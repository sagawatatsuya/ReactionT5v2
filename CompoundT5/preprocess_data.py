import subprocess
import sys
import warnings

import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split

sys.path.append("../")
from utils import remove_atom_mapping, seed_everything

seed_everything(seed=42)

# Disable RDKit warnings and Python warnings
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

files_to_download = [
    "1ZPsoUYb4HcxFzK_ac9rb_pQj7oO3Gagh",
    "1XwkxxHiaWFbSNhGyxnv6hAliutIMNrIp",
    "1yIwUH_OhER9nuMo9HjBhBmyc6zvmrSPA",
    "1skFRirstIUijhieshvJEScBD2aB3H1YU",
    "1fa2MyLdN1vcA7Rysk8kLQENE92YejS9B",
]

for file_id in files_to_download:
    subprocess.run(
        f"gdown 'https://drive.google.com/uc?export=download&id={file_id}'", shell=True
    )

# Move downloaded files to data directory
subprocess.run("mv *.smi ../data", shell=True)
subprocess.run("mv *.tsv ../data", shell=True)


# Function to process SMILES files and save canonicalized versions
def process_smiles_files(file_paths):
    unique_smiles = set()
    for file_path in file_paths:
        suppl = Chem.SmilesMolSupplier(file_path)
        for mol in suppl:
            if mol is not None:
                try:
                    sm = Chem.MolToSmiles(mol, canonical=True)
                    unique_smiles.add(sm)
                except:
                    continue
    df = pd.DataFrame({"smiles": list(unique_smiles)})
    df.to_csv("../data/ZINC-canonicalized.csv", index=False)

    train, valid = train_test_split(df, test_size=0.1)
    # Save train and validation data
    train.to_csv("../data/ZINC-canonicalized-train.csv", index=False)
    valid.to_csv("../data/ZINC-canonicalized-valid.csv", index=False)


# Process 16_p files
process_smiles_files([f"../data/16_p{i}.smi" for i in range(4)])


# Load reaction data
ord_df = pd.read_csv(
    "../data/all_ord_reaction_uniq_with_attr20240506_v1.tsv",
    sep="\t",
    names=["id", "input", "product", "condition"],
)


def data_split(row):
    categories = [
        "CATALYST",
        "REACTANT",
        "REAGENT",
        "SOLVENT",
        "INTERNAL_STANDARD",
        "NoData",
    ]
    data = {cat: [] for cat in categories}
    input_data = row["input"]

    if isinstance(input_data, str):
        for item in input_data.split("."):
            for cat in categories:
                if cat in item:
                    data[cat].append(item[item.find(":") + 1 :])
                    break

    for key, value in data.items():
        data[key] = ".".join(value)

    product_data = row["product"]
    if isinstance(product_data, str):
        product_data = product_data.replace(".PRODUCT", "PRODUCT")
        pro_lis = []
        for item in product_data.split("PRODUCT:"):
            if item != "":
                pro_lis.append(item)
        data["PRODUCT"] = ".".join(pro_lis)
    else:
        data["PRODUCT"] = None

    condition_data = row["condition"]
    if isinstance(condition_data, str):
        data["YIELD"] = (
            float(condition_data.split(":")[1]) if "YIELD" in condition_data else None
        )
        temp_pos = condition_data.find("TEMP")
        data["TEMP"] = (
            float(condition_data[temp_pos:].split(":")[1])
            if "TEMP" in condition_data
            else None
        )
    else:
        data["YIELD"] = None
        data["TEMP"] = None

    return list(data.values())


# Split data and create cleaned DataFrame
categories = [
    "CATALYST",
    "REACTANT",
    "REAGENT",
    "SOLVENT",
    "INTERNAL_STANDARD",
    "NoData",
    "PRODUCT",
    "YIELD",
    "TEMP",
]
cleaned_data = {cat: [] for cat in categories}

for _, row in ord_df.iterrows():
    split_data = data_split(row)
    for i, value in enumerate(split_data):
        cleaned_data[categories[i]].append(value)

cleaned_df = pd.DataFrame(cleaned_data)

# Apply remove_atom_mapping function to relevant columns
for column in [
    "CATALYST",
    "REACTANT",
    "REAGENT",
    "SOLVENT",
    "INTERNAL_STANDARD",
    "NoData",
    "PRODUCT",
]:
    cleaned_df[column] = cleaned_df[column].apply(
        lambda x: remove_atom_mapping(x) if isinstance(x, str) else None
    )

# Save cleaned DataFrame
cleaned_df.to_csv("../data/preprocessed_ord.tsv", index=False)

train, valid = train_test_split(cleaned_df, test_size=int(len(cleaned_df) * 0.1))
train, test = train_test_split(train, test_size=int(len(cleaned_df) * 0.1))
# Save train and validation data
train.to_csv("../data/preprocessed_ord_train.csv", index=False)
valid.to_csv("../data/preprocessed_ord_valid.csv", index=False)
test.to_csv("../data/preprocessed_ord_test.csv", index=False)
