# ReactionT5v2
ReactionT5 is a T5 model pretrained on a vast array of chemical reactions from the [Open Reaction Database (ORD)](https://github.com/open-reaction-database/ord-data). Unlike other models that are trained on smaller, potentially biased datasets (e.g. patent datasets or high-throughput reaction datasets created by a single reaction), ReactionT5 leverages the extyensive and diverse dataset provided by ORD. This ensures greater generalizability and performance, enabling ReactionT5 to condact product, retrosynthesis, and yield prediction of unseen chemical reactions with high accuracy. This makes it highly practical for real-world applications.

![model image](https://github.com/sagawatatsuya/ReactionT5/blob/main/model-image.png)


In this repository, we will demonstrate how to use ReactionT5 for product prediction, retrosynthesis prediction, and yield prediction on your own datasets. The pretrained models, datasets, and demo is available at [Hugging Face Hub](https://huggingface.co/sagawa).


- [ReactionT5v2](#reactiont5v2)  
  - [Setup](#setup)  
  - [Usage](#usage)  
  - [Fine-tuning](#fine-tuning)  
  - [Structure](#structure) 
  - [Authors](#authors)
  - [Citation](#citation)  


# Setup
ReactionT5 is based on the transformers library. Additionally, RDKit is used for validity check of predicted compounds. To install these and other necessary libraries, use the following commands:
```
pip install rdkit
pip install pytorch
pip install tokenizers==0.12.1
pip install transformers==4.21.0
pip install datasets
pip install sentencepiece==0.1.96
```


# Usage
You can use ReactionT5 for product prediction, retrosynthesis prediction, and yield prediction.

### Task: Forward
To predict the products of reactions from their inputs, use the following command. The code expects 'input_data' as a string or CSV file that contains an 'input' column. The format of the string or the contents of the column should follow this template: "REACTANT:{SMILES of reactants}REAGENT:{SMILES of reagents, catalysts, or solvents}". If there are no catalyst, reagent, or solvents, fill the blank with a space. For multiple compounds, concatenate them with ".".(ex. "REACTANT:COC(=O)C1=CCCN(C)C1.O.\[Al+3].\[H-].\[Li+].\[Na+].\[OH-]REAGENT:C1CCOC1")
```
cd task_forward
python prediction.py \
    --input_data="../data/task_forward_demo_input.csv" \
    --input_max_length=150 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=16 \
    --output_dir="output"
```

### Task: Retrosynthesis
To predict the reactants of reactions from their products, use the following command. The code expects 'input_data' as a string or CSV file that contains an 'input' column. The format of the string or the contents of the column should be SMILES of products. For multiple compounds, concatenate them with ".".(ex. "CCN(CC)CCNC(=S)NC1CCCc2cc(C)cnc21")
```
cd task_retrosynthesis
python prediction.py \
    --input_data="../data/task_retrosynthesis_demo_input.csv" \
    --input_max_length=100 \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=16 \
    --output_dir="output"
```

### Task: Yield
To predict the yields of reactions from their inputs, use the following command. The code expects 'input_data' as a string or CSV file that contains an 'input' column. The format of the string or the contents of the column should follow this template: "REACTANT:{SMILES of reactants}REAGENT:{SMILES of reagents, catalysts, or solvents}PRODUCT:{SMILES of products}". If there are multiple compounds, concatenate them with ".".(ex. "REACTANT:C1CCNCC1.CC1(C)OC(=O)C(Oc2ccc(Br)cn2)=C1c1ccc(S(C)(=O)=O)cc1REAGENT:C1CN2CCN1CC2.COCCOC.Cl[Ni]ClPRODUCT:COC(=O)CC1CCc2cc(N3CCCCC3)cc3[nH]c(=O)c(=O)n1c23")
```
cd task_yield
python prediction_with_PreTrainedModel.py \
    --data="../data/task_yield_demo_input.csv" \
    --batch_size=16 \
    --output_dir="output"
```


# Fine-tuning
If your dataset is very specific and different from ORD's data, ReactionT5 may not predict well. In that case, you can fine-tune ReactionT5 on your dataset. From our study, ReactionT5's performance drastically improved its performance by fine-tuning using relatively small data (100 reactions).

### Task: Forward
Specify your training and validation data used for fine-tuning and run the following command. The code expects these data to contain columns named 'REACTANT', 'REAGENT', and 'PRODUCT'; each has SMILES information. If there is no reagent information, fill in the blank with ' '.
```
cd task_forward
python finetune.py \
    --epochs=50 \
    --batch_size=32 \
    --train_data_path='../data/demo_reaction_data.csv' \
    --valid_data_path='../data/demo_reaction_data.csv' \
    --output_dir='output'
```

### Task: Retrosynthesis
Specify your training and validation data used for fine-tuning and run the following command. The code expects these data to contain columns named 'REACTANT' and 'PRODUCT'; each has SMILES information. If there is no reagent information, fill in the blank with ' '.
```
cd task_retrosynthesis
python finetune.py \
    --epochs=20 \
    --batch_size=32 \
    --train_data_path='../data/demo_reaction_data.csv' \
    --valid_data_path='../data/demo_reaction_data.csv' \
    --output_dir='output'
```

### Task: Yield
Specify your training and validation data used for fine-tuning and run the following command. We expect these data to contain columns named 'REACTANT', 'REAGENT', 'PRODUCT', and 'YIELD'; except 'YIELD ' have SMILES information, and 'YIELD' has numerical information. If there is no reagent information, fill in the blank with ' '.
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

# Structure
```
ReactionT5v2/  
├── CompoundT5/                     # Codes used for data processing and compound pretraining  
│ ├── README.md                     # More detailed README file for CompoundT5 creation  
│ └── requirements.yaml             # Required Packages for compound pretraining  
├── data/                           # Datasets  
├── multitask_no_pretraining/       # Multitask (forward, retrosynthesis, and yield prediction and finetuning)  
├── multitask_pretraining/          # Multitask (forward, retrosynthesis, and yield prediction and finetuning)  
├── task_forward/                   # Forward prediction and finetuning  
├── task_retrosynthesis/            # Retrosynthesis prediction and finetuning  
├── task_yield/                     # Yield prediction and finetuning  
├── requirements.yaml               # Required packages for prediction and finetuning  
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