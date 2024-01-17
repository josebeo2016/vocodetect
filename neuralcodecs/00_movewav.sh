#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2

# list all wav files in the current directory
filelist=$(find $INPUT_DIR -maxdepth 1 -name "*.flac")
filecount=$(find $INPUT_DIR -maxdepth 1 -name "*.flac" | wc -l)
counter=0
for file in $filelist
do
    # get the filename without the path
    filename=$(basename $file)
    # replace .flac in the filename with .wav
    newfilename=${filename/.flac/.wav}
    # convert flac to wav
    ffmpeg -i $file $OUTPUT_DIR/$newfilename.wav > /dev/null 2>&1
    counter=$((counter+1))
    percent=$((100*counter/filecount))
    printf "\rProgress: [%-50s] %d%%, processed %d out of %d files." $(printf '%.0s#' $(seq 1 $((percent/2)))) $percent $counter $filecount
done