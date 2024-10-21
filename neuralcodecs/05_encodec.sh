#!/bin/bash
INPUT_DIR=$1
OUTPUT_DIR=$2

TARGET_BANDWIDTH=6

filelist=$(find $INPUT_DIR -maxdepth 1 -name "*.wav")
filecount=$(find $INPUT_DIR -maxdepth 1 -name "*.wav" | wc -l)
counter=0
for file in $filelist
do
    filename=$(basename -- "$file")
    OUTPUT_WAV_FILE=$OUTPUT_DIR/$filename
    encodec -r -b $TARGET_BANDWIDTH -f $file $OUTPUT_WAV_FILE
    counter=$((counter+1))
    percent=$((100*counter/filecount))
    printf "\rProgress: [%-50s] %d%%, processed %d out of %d files." $(printf '%.0s#' $(seq 1 $((percent/2)))) $percent $counter $filecount
done