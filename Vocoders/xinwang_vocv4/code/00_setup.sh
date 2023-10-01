#!/bin/bash

# We need scripts and data IOs in this repo
git clone --depth 1 https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts.git

# install conda environment
cd project-NN-Pytorch-scripts
conda env create -f env.yml

# load conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch-1.7

# install pyworld
# we use pyworld for F0 extraction
pip install pyworld==0.3.0
