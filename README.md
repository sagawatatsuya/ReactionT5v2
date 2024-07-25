# ReactionT5
ReactionT5 is a T5 model pretrained on a large amount of chemical reactions in the [Open Reaction Database (ORD)](https://github.com/open-reaction-database/ord-data). Unlike other models for chemical reaction prediction that are trained on small and potentially biased datasets (e.g. patent datasets or high-throughput reaction datasets created by a single reaction), ReactionT5 leverages the vast and diverse dataset provided by ORD to ensure greater generalizability and performance. This allows ReactionT5 to predict both the products and yields of unseen chemical reactions with high accuracy, making it highly practical for real-world applications.

![model image](https://github.com/sagawatatsuya/ReactionT5/blob/main/model-image.png)


In this repository, we will show how to use ReactionT5 for product prediction and yield prediction on your own datasets. The pretrained models, datasets, and demo is available at [Hugging Face Hub](https://huggingface.co/sagawa).


# Installation
Reaction T5 is based on the transformers library. In addition, RDKit is used for validity check of predicted products. To install these and other necessary libraries, use the following commands:
```
pip install rdkit
pip install pytorch
pip install tokenizers==0.12.1
pip install transformers==4.21.0
pip install datasets
pip install sentencepiece==0.1.96
```


# Use ReactionT5
You can use ReactionT5 to predict the products and yields of chemical reactions.

### Product prediction
To predict the products of reactions from their inputs, use the following command. The code expects 'input_data' as a string or CSV file that contains an 'input' column. The format of the string or contents of the column should follow this template: "REACTANT:{SMILES of reactants}REAGENT:{SMILES of reagents, catalysts, or solvents}". If there are no catalyst, reagent, or solvents, fill the blank with a space. And if there are multiple compounds, concatenate them with ".".(ex. "REACTANT:COC(=O)C1=CCCN(C)C1.O.\[Al+3].\[H-].\[Li+].\[Na+].\[OH-]REAGENT:C1CCOC1")
```
cd forward_reaction_prediction/
python prediction.py \
    --input_data="../data/forward_reaction_prediction_demo_input.csv" \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=16 \
    --output_dir="./"
```

### Yield prediction
To predict the yields of reactions from their inputs, use the following command. The code expects 'input_data' as a string or CSV file that contains an 'input' column. The format of the string or contents of the column should follow this template: "REACTANT:{SMILES of reactants}REAGENT:{SMILES of reagents, catalysts, or solvents}PRODUCT:{SMILES of products}". If there are multiple compounds, concatenate them with ".".(ex. "REACTANT:CC(C)n1ncnc1-c1cn2c(n1)-c1cnc(O)cc1OCC2.CCN(C(C)C)C(C)C.Cl.NC(=O)\[C@@H]1C\[C@H](F)CN1REAGENT: PRODUCT:O=C(NNC(=O)C(F)(F)F)C(F)(F)F")
When running the command for the first time, you should include the 'download_pretrained_model' argument.
```
cd yield_prediction/
python prediction.py \
    --data="../data/yield_prediction_demo_input.csv" \
    --batch_size=10 \
    --output_dir="./" \
    --download_pretrained_model
```


# Fine-tuning
If your dataset is very specific and different from ORD's data, ReactionT5 may not predict well. In that case, you can conduct fine-tuning of ReactionT5 on your dataset. From our study, ReactionT5's performance drastically improved its performance by fine-tuning using relatively small data (200 reactions).

### Product prediction
Specify your training and validation data used for fine-tuning and run the following command. We expect these data to contain columns named 'REACTANT', 'REAGENT', and 'PRODUCT'; each has SMILES information. If there is no reagent information, fill in the blank with ' '.
```
cd forward_reaction_prediction/
pip install sacrebleu
python finetune-pretrained-ReactionT5.py \
    --epochs=50 \
    --batch_size=32 \
    --train_data_path='your_train.csv' \
    --valid_data_path='your_validation.csv'
```

### Yield prediction
Specify your training and validation data used for fine-tuning and run the following command. We expect these data to contain columns named 'REACTANT', 'REAGENT', 'PRODUCT', and 'YIELD'; except 'YIELD ' have SMILES information, and 'YIELD' has numerical information. If there is no reagent information, fill in the blank with ' '.
```
python finetuning.py \
    --epochs=200 \
    --batch_size=6 \
    --train_data_path='your_train.csv' \
    --valid_data_path='your_validation.csv' \
    --download_pretrained_model \
    --output_dir='output/'
```

## Citation
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