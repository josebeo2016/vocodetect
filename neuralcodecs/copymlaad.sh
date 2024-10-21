#!/bin/bash
INPUT_DIR=$1
OUTPUT_DIR=$2
filelist=$(find $INPUT_DIR -maxdepth 6 -name "*.wav")
filecount=$(find $INPUT_DIR -maxdepth 6 -name "*.wav" | wc -l)
counter=0
for file in $filelist
do
    # use ffmpeg to convert the audio to 16kHz mono PCM
    filename=$(basename -- "$file")
    ffmpeg -i $file -ac 1 -ar 16000 -f wav $OUTPUT_DIR/$filename -y > /dev/null 2>&1
    # cp $file $OUTPUT_DIR
    counter=$((counter+1))
    percent=$((100*counter/filecount))
    printf "\rProgress: [%-50s] %d%%, processed %d out of %d files." $(printf '%.0s#' $(seq 1 $((percent/2)))) $percent $counter $filecount
done