#!/bin/bash
# Install dependency for fairseq

# Name of the conda environment
# ENVNAME=fairseq

# pyenv activate $ENVNAME

pip install pip==24.0

# echo "===========Install pytorch==========="
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# install flask
pip install flask

# install scipy
pip install scipy==1.7.3

# install gunicorn
pip install gunicorn==23.0.0

# install pandas
pip install pandas==1.3.5

# install protobuf
pip install protobuf==3.20.3

# install matplotlib
pip install matplotlib==3.7.1

# install tensorboard
pip install tensorboard==2.6.0
pip install tensorboardX==2.6

# install librosa
pip install librosa==0.10.0

# install auditok
pip install auditok==0.2.0

# install demucs
pip install demucs==4.0.0

# install PyWavelets
pip install PyWavelets==1.4.1

# install h5py
pip install h5py==3.7.0

# install spafe
pip install spafe==0.1.2

# install pydub
pip install pydub==0.25.1

# install pyyaml
pip install pyyaml

# install tqdm
pip install tqdm

# install asteroid-filterbanks
pip install asteroid-filterbanks

# install einops
pip install einops

# install fairseq
pip install git+https://github.com/facebookresearch/fairseq.git@a54021305d6b3c4c5959ac9395135f63202db8f1

# install transformers
pip install transformers

# install mamba_ssm for SEMamba. If conflict then remove the denoise mamba later
pip install mamba_ssm==1.2.0_post1

# install peft==0.15.1
pip install peft==0.15.1

# install psola
pip install psola

# install kaldiio
pip install kaldiio==2.18.1

# install faiss
pip install faiss==1.7.3
pip install faiss-gpu==1.7.2
