# ReactionT5v2
ReactionT5 is a T5 model pretrained on a vast array of chemical reactions from the [Open Reaction Database (ORD)](https://github.com/open-reaction-database/ord-data). Unlike other models that are trained on smaller, potentially biased datasets (e.g. patent datasets or high-throughput reaction datasets created by a single reaction), ReactionT5 leverages the extyensive and diverse dataset provided by ORD. This ensures greater generalizability and performance, enabling ReactionT5 to condact product, retrosynthesis, and yield prediction of unseen chemical reactions with high accuracy. This makes it highly practical for real-world applications.

![model image](https://github.com/sagawatatsuya/ReactionT5v2/blob/main/model-image.png)


In this repository, we will demonstrate how to use ReactionT5 for product prediction, retrosynthesis prediction, and yield prediction on your own datasets. The pretrained models and demo is available at [Hugging Face Hub](https://huggingface.co/collections/sagawa/reactiont5-67dbe0550cbb6886a85e348b).

# Table of Contents
- [ReactionT5v2](#reactiont5v2)  
  - [Setup](#setup)  
  - [Dataset](#dataset)
  - [Usage](#usage)  
  - [Fine-tuning](#fine-tuning)  
  - [Compound Pre-training](#compound-pre-training)  
  - [Reaction Pre-training](#reaction-pre-training)  
  - [Structure](#structure) 
  - [Authors](#authors)
  - [Citation](#citation)  


# Setup
ReactionT5 is based on the transformers library. Additionally, RDKit is used for validity check of predicted compounds. To install these and other necessary libraries, use the following commands:
```
pip install rdkit
pip install torch
pip install tokenizers==0.19.1
pip install transformers==4.40.2
pip install datasets
pip install accelerate -U
pip install sentencepiece
```

# Dataset
For model training and finetuning, we used the ORD dataset, USPTO_MIT dataset, USPTO_50k dataset, and C-N cross-coupling reactions dataset. Each dataset can be downloaded from the following links:
- [ORD](https://drive.google.com/file/d/1JozA2OlByfZ-ILt5H5YrTjLJvSvD8xdL/view?usp=drive_link)
- [USPTO_MIT](https://drive.google.com/file/d/1NEb5DSwqFfg4gm0P2lPb19cqPt96-OSC/view?usp=drive_link)
- [USPTO_50k](https://drive.google.com/file/d/15-E4eaxsUJ_aKxX0mnOrvYCTWKpqrLvf/view?usp=drive_link)
- [Buchwald-Hartwig C-N cross-coupling](https://drive.google.com/file/d/1qvsJk1_q1yfJvBYtEMGOcCuk2pRvsWvy/view?usp=drive_link)


# Usage
You can use ReactionT5 for product prediction, retrosynthesis prediction, and yield prediction.

### Task: Forward
To predict the products of reactions from their inputs, use the following command. The code expects 'input_data' as a CSV file that contains columns named 'REACTANT', 'REAGENT', and 'PRODUCT'; each has SMILES information. If there is no reagent information, fill in the blank with ' '. For multiple compounds, concatenate them with ".".
```
cd task_forward
python prediction.py \
    --input_data="../data/demo_reaction_data.csv" \
    --model_name_or_path="sagawa/ReactionT5v2-forward" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=16 \
    --output_dir="output"
```

### Task: Retrosynthesis
To predict the reactants of reactions from their products, use the following command. The code expects 'input_data' as a CSV file that contains a "PRODUCT" column. The format of the contents of the column should be SMILES of each compound. For multiple compounds, concatenate them with ".".
```
cd task_retrosynthesis
python prediction.py \
    --input_data="../data/demo_reaction_data.csv" \
    --model_name_or_path="sagawa/ReactionT5v2-retrosynthesis" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=16 \
    --output_dir="output"
```

### Task: Yield
To predict the yields of reactions from their inputs, use the following command. The code expects 'input_data' as a CSV file that contains columns named 'REACTANT', 'REAGENT', 'PRODUCT', and 'YIELD'; except 'YIELD' have SMILES information, and 'YIELD' has numerical information. If there is no reagent information, fill in the blank with ' '. For multiple compounds, concatenate them with ".".
```
cd task_yield
python prediction_with_PreTrainedModel.py \
    --input_data="../data/demo_reaction_data.csv" \
    --model_name_or_path="sagawa/ReactionT5v2-yield" \
    --batch_size=16 \
    --output_dir="output"
```


# Fine-tuning
If your dataset is very specific and different from ORD's data, ReactionT5 may not predict well. In that case, you can fine-tune ReactionT5 on your dataset. From our study, ReactionT5's performance drastically improved its performance by fine-tuning using relatively small data (100 reactions).

### Task: Forward
Specify your training and validation data used for fine-tuning and run the following command. The code expects train and valid data that contain columns named 'REACTANT', 'REAGENT', and 'PRODUCT'; each has SMILES information. If there is no reagent information, fill in the blank with ' '. For multiple compounds, concatenate them with ".".
```
cd task_forward
python finetune.py \
    --model_name_or_path='sagawa/ReactionT5v2-forward' \
    --epochs=50 \
    --batch_size=32 \
    --train_data_path='../data/demo_reaction_data.csv' \
    --valid_data_path='../data/demo_reaction_data.csv' \
    --output_dir="output"
```

### Task: Retrosynthesis
Specify your training and validation data used for fine-tuning and run the following command. The code expects train and valid data that contain columns named 'REACTANT' and 'PRODUCT'; each has SMILES information. For multiple compounds, concatenate them with ".".
```
cd task_retrosynthesis
python finetune.py \
    --model_name_or_path='sagawa/ReactionT5v2-retrosynthesis' \
    --epochs=20 \
    --batch_size=32 \
    --train_data_path='../data/demo_reaction_data.csv' \
    --valid_data_path='../data/demo_reaction_data.csv' \
    --output_dir='output'
```

### Task: Yield
Specify your training and validation data used for fine-tuning and run the following command. The code expects train and valid data that contain columns named 'REACTANT', 'REAGENT', 'PRODUCT', and 'YIELD'; except 'YIELD ' have SMILES information, and 'YIELD' has numerical information. If there is no reagent information, fill in the blank with ' '. For multiple compounds, concatenate them with ".".
```
cd task_yield
python finetune.py \
    --epochs=200 \
    --batch_size=32 \
    --train_data_path='../data/demo_reaction_data.csv' \
    --valid_data_path='../data/demo_reaction_data.csv' \
    --download_pretrained_model \
    --output_dir='output'
```

# Compound Pre-training
If you want to retrain CompoundT5 (compound pre-training), please refer to the [README.md](./CompoundT5/README.md) file located inside the CompoundT5 directory.


# Reaction Pre-training
If you want to retrain ReactionT5 from CompoundT5 (reaction pre-training), you can do so by running the following commands. These will train ReactionT5 on the Open Reaction Database (ORD) dataset.

### Dataset
Download the preprocessed ORD dataset from [here](https://drive.google.com/file/d/1JozA2OlByfZ-ILt5H5YrTjLJvSvD8xdL/view?usp=drive_link) and place it in the data/ORD directory.  
To prevent data leakage in downstream benchmarks, we specify either 'USPTO_test_data_path' or 'CN_test_data_path' in the training commands below and remove any duplicated reactions from the training data. You can also train ReactionT5 on full ORD data without specifying these parameters.  
The corresponding test datasets can be downloaded from the following links:
- [USPTO_MIT](https://drive.google.com/file/d/1NEb5DSwqFfg4gm0P2lPb19cqPt96-OSC/view?usp=drive_link)
- [USPTO_50k](https://drive.google.com/file/d/15-E4eaxsUJ_aKxX0mnOrvYCTWKpqrLvf/view?usp=drive_link)
- [Buchwald-Hartwig C-N cross-coupling](https://drive.google.com/file/d/1qvsJk1_q1yfJvBYtEMGOcCuk2pRvsWvy/view?usp=drive_link)

### Task: Forward
```
cd task_forward
python train.py \
    --output_dir='ReactionT5_forward' \
    --epochs=100 \
    --lr=1e-3 \
    --batch_size=32 \
    --input_max_len=150 \
    --target_max_len=100 \
    --weight_decay=0.01 \
    --evaluation_strategy='epoch' \
    --save_strategy='epoch' \
    --logging_strategy='epoch' \
    --save_total_limit=100 \
    --train_data_path='../data/ORD/all_ord_reaction_uniq_with_attr20240506_v3_train.csv' \
    --valid_data_path='../data/ORD/all_ord_reaction_uniq_with_attr20240506_v3_valid.csv' \
    --test_data_path='../data/ORD/all_ord_reaction_uniq_with_attr20240506_v3_test.csv' \
    --USPTO_test_data_path='../data/USPTO_MIT/MIT_separated/test.csv' \
    --disable_tqdm \
    --pretrained_model_name_or_path='sagawa/CompoundT5'
```
### Task: Retrosynthesis
```
cd task_retrosynthesis
python train.py \
    --output_dir='ReactionT5_retrosynthesis' \
    --epochs=100 \
    --lr=2e-4 \
    --batch_size=32 \
    --input_max_len=100 \
    --target_max_len=150 \
    --weight_decay=0.01 \
    --evaluation_strategy='epoch' \
    --save_strategy='epoch' \
    --logging_strategy='epoch' \
    --save_total_limit=100 \
    --train_data_path='../data/ORD/all_ord_reaction_uniq_with_attr20240506_v3_train.csv' \
    --valid_data_path='../data/ORD/all_ord_reaction_uniq_with_attr20240506_v3_valid.csv' \
    --test_data_path='../data/ORD/all_ord_reaction_uniq_with_attr20240506_v3_test.csv' \
    --USPTO_test_data_path='../data/USPTO_50k/test.csv' \
    --disable_tqdm \
    --pretrained_model_name_or_path='sagawa/CompoundT5'
```
### Task: Yield
```
cd task_yield
python train.py \
    --output_dir='ReactionT5_yield_CN_test1' \
    --train_data_path='../data/ORD/all_ord_reaction_uniq_with_attr20240506_v3_train.csv' \
    --valid_data_path='../data/ORD/all_ord_reaction_uniq_with_attr20240506_v3_valid.csv' \
    --test_data_path='../data/ORD/all_ord_reaction_uniq_with_attr20240506_v3_test.csv' \
    --CN_test_data_path='../data/C_N_yield/MFF_Test1/test.csv' \
    --epochs=100 \
    --input_max_length=300 \
    --pretrained_model_name_or_path='sagawa/CompoundT5' \
    --batch_size=32
```


# Structure
```
ReactionT5v2/  
├── CompoundT5/                     # Codes used for data processing and compound pretraining  
│ └── README.md                     # More detailed README file for CompoundT5 creation  
├── data/                           # Datasets  
├── task_forward/                   # Forward prediction and finetuning  
├── task_retrosynthesis/            # Retrosynthesis prediction and finetuning  
├── task_yield/                     # Yield prediction and finetuning  
└── README.md                       # This README file  
```


# Authors
Tatsuya Sagawa, Ryosuke Kojima

# Citation
arxiv link: https://arxiv.org/abs/2311.06708
```
@misc{sagawa2023reactiont5,  
      title={ReactionT5: a large-scale pre-trained model towards application of limited reaction data}, 
      author={Tatsuya Sagawa and Ryosuke Kojima},  
      year={2023},  
      eprint={2311.06708},  
      archivePrefix={arXiv},  
      primaryClass={physics.chem-ph}  
}
```