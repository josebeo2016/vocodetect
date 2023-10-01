#!/usr/bin/bash
# Usage: 
#  1. Please specify the input waveform directory
#     Please specify a path to save the file list
#     Please specify a directory to save acoustic features
#
#  2. bash 01_feature_extraction.sh

PRJDIR=$PWD

# ========== configurations =========
# input waveform directory
# here, we use a toy subset from asvspoof19_bona for demonstration
# you may also try $PRJDIR/../data/voxceleb2/dev
INPUTWAVDIR=$PRJDIR/../../traindata/CNSL_intern/norm/

# output feature directory
OUTFEATDIR=$PRJDIR/../data-feat/cnsl_intern/

# path to save a file list
SCPLIST=$OUTFEATDIR/file.lst
# ===================================

# load environment
eval "$(conda shell.bash hook)"
conda activate py39
export PYTHONPATH=$PWD/project-NN-Pytorch-scripts:${PYTHONPATH}

mkdir -p ${OUTFEATDIR}

echo "Processing ${INPUTWAVDIR}"

# get file list
cd ${INPUTWAVDIR}
find ./ -type f -name "*.wav" | sed "s:^./::g" | sed "s:\.wav$::g" > ${SCPLIST}
cd ${PRJDIR}

# Run code
python feature_extraction.py ${SCPLIST} ${INPUTWAVDIR} ${OUTFEATDIR}
echo "Acoustic features saved to ${OUTFEATDIR}"
echo "File list saved to ${SCPLIST}"

# Done
