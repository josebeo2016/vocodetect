#!/bin/bash
# Please change the configurations below, then
# bash 02_vocoding.sh
#
# Here, we use hifigan and do copy-synthesis on a toy subset of voxceleb2/dev 

# ==== configurations
# which training set has been used to train the model?
# it can be pretrained_voxceleb2, pretrained_asvspoof_trn_bona, 
# pretrained_libritts_asvspoof_trn_bona, or pretrained_libritts_subset
trnset=pretrained_libritts_asvspoof_trn_bona

# model name, hifigan, hn-sinc-nsf, hn-sinc-nsf-hifi, or waveglow
modelname=hn-sinc-nsf
# name of the pre-trained model file
#  for GAN-based models (hifigan and hn_sinc_nsf_hifi)
# trainedmodel=trained_network_G.pt
#  for others, it is trained_network.pt
trainedmodel=trained_network.pt

# input:
# path to the input features of the data set to be vocoded
# ${OUTFEATDIR} in 01_feature_extraction.sh
featdir=$PWD/../data-feat/cnsl_intern/
# assign an arbitary name to the data set (it should not contain '/')
setname=dataset_cnsl_intern
# file list, generated by 01_feature_extraction.sh
filelist=${featdir}/file.lst

# output:
# path to save the output waveform
outputdir=$PWD/../../traindata/CNSL_intern/vocoded/${modelname}


# folder of the pre-trained model
prjdir=$PWD/../${trnset}/${modelname}

# a temporary cache directory
cachepath=$PWD/../cache
# ====================

# load conda environment and add path
eval "$(conda shell.bash hook)"
conda activate py39
export PYTHONPATH=$PWD/project-NN-Pytorch-scripts:${PYTHONPATH}


if [ ! -e ${filelist} ];
then
    echo "Cannot find ${filelist}"
else
    
    # a cache directory to save data length
    if [ ! -d ${cachepath} ];
    then
	mkdir ${cachepath}
    fi
    
    # export environment variables
    # they will be loaded by the config.py in each folder
    # not every model uses F0
    export TEMP_TESTSET_NAME=${setname}
    export TEMP_TESTSET_LST=${filelist}
    export TEMP_TESTSET_F0=${featdir}
    export TEMP_TESTSET_MEL=${featdir}

    cd ${prjdir}

    com="python main.py --inference --trained-model ${trainedmodel} \
            --cudnn-deterministic-toggle \
            --output-dir ${outputdir} --path-cache-file ${cachepath}"
    echo ${com}
    eval ${com}
fi