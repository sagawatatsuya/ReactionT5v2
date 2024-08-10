# ReactionT5
![training_procedure_image](https://github.com/sagawatatsuya/ReactionT5/blob/main/study_reproduction/training-procedure.png)
Here, we will explain how to do compound pretraining. 

# Installation
To get started, you will first need to install the necessary libraries. You can use the requirements.yaml file for this purpose. If the versions of torch and jax do not match your environment, you can change and run the following command:
```
conda install -c conda-forge rdkit
conda install -c conda-forge gdown
conda install scikit-learn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tokenizers==0.12.1
pip install transformers==4.21.0
pip install datasets
pip install sentencepiece==0.1.96
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax
conda install -c conda-forge optuna
```
This will install all the necessary libraries for the project.

The original data used for this study is uploaded to Google Drive and can be found at the following links:
・[ZINC](https://drive.google.com/drive/folders/1SgM35D14JUqgNILxaiRQYbZoyooFOF-3)  
・[ORD](https://drive.google.com/file/d/1Qbsl8_CmdIK_iNNY8F6wATVnDQNSW9Tc/view?usp=drive_link)  
The pre-processed data is also available on [Hugging Face Hub](https://huggingface.co/sagawa) and can be used directly. 

To download the data, you can run the following command:
```
python preprocess_data.py
```
To complete the preparation for compound pretraining, run the following command:
```
python prepare_model.py
```

# Compound pretraining
Run the following command to conduct compound pretraining. In compound pretraining, T5 is trained on the ZINC dataset using span-masked language modeling. The pretraine model (CompoundT5) is uploaded to [Hugging Face Hub](https://huggingface.co/sagawa/CompoundT5).
```
cd CompoundT5
sh run.sh
```
Please note that if your GPU memory size is small, you may encounter an out-of-memory error during T5 pre-training. If this occurs, you can try reducing the batch size or you can try putting XLA_PYTHON_CLIENT_MEM_FRACTION=.8 before python ./new_run_t5_mlm_flax.py in run.sh file. This reduces GPU memory preallocation.