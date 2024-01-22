#!/bin/bash

# This script is used filter the bonafide audio and copy to la2019_bona

TRAIN_LA=/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_train/lyra3200/
DEV_LA=/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_dev/lyra3200/

OUTPUT_DIR=./la2019_bona/

# create the output directory if it does not exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Train bona

filelist=$(awk '{print $1}' /dataa/phucdt/vocodetect/traindata/asvspoof_2019_supcon/scp/train_bonafide.lst)
filecount=$(awk '{print $1}' /dataa/phucdt/vocodetect/traindata/asvspoof_2019_supcon/scp/train_bonafide.lst | wc -l)

counter=0
for file in $filelist
do
    filename=${file/.wav/_decoded.wav}

    cp $TRAIN_LA/$filename $OUTPUT_DIR/lyra_$file
    counter=$((counter+1))
    percent=$((100*counter/filecount))
    printf "\rProgress: [%-50s] %d%%, processed %d out of %d files." $(printf '%.0s#' $(seq 1 $((percent/2)))) $percent $counter $filecount
done

# Dev bona

filelist=$(awk '{print $1}' /dataa/phucdt/vocodetect/traindata/asvspoof_2019_supcon/scp/dev_bonafide.lst)
filecount=$(awk '{print $1}' /dataa/phucdt/vocodetect/traindata/asvspoof_2019_supcon/scp/dev_bonafide.lst | wc -l)

counter=0
for file in $filelist
do
    filename=${file/.wav/.wav_decoded.wav}

    cp $DEV_LA/$filename $OUTPUT_DIR/lyra_$file
    counter=$((counter+1))
    percent=$((100*counter/filecount))
    printf "\rProgress: [%-50s] %d%%, processed %d out of %d files." $(printf '%.0s#' $(seq 1 $((percent/2)))) $percent $counter $filecount
done