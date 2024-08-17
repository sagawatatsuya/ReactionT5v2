conda create -n deepspeed python=3.11
conda activate deepspeed
conda install -c conda-forge mpi4py openmpi
conda install -c conda-forge cudatoolkit-dev
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install scikit-learn
pip install transformers
pip install datasets
pip install accelerate -U
pip install --upgrade deepspeed
