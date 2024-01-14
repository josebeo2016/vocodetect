#!/bin/bash

# This script encodes the audio files in the specified directory using the Lyra encoder.
# Usage: ./01_lyra_encoder.sh <path_to_audio_files> <path_to_output_directory>

INPUT_DIR=$1
OUTPUT_DIR=$2
BITRATE=3200
# check if the number of arguments is correct
if [ $# -ne 2 ]; then
    echo "Usage: ./01_lyra_encoder.sh <path_to_audio_files> <path_to_output_directory>"
    exit 1
fi

# check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory does not exist."
    exit 1
fi

# create the output directory if it does not exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

filelist=$(ls $INPUT_DIR/*.wav)
filecount=$(ls $INPUT_DIR/*.wav | wc -l)
# encode the audio files
cd lyra/
counter=0
for file in $filelist
do
    # echo "encoding $file"
    bazel-bin/lyra/cli_example/encoder_main --input_path $file --output_dir $OUTPUT_DIR --bitrate=$BITRATE > /dev/null 2>&1
    counter=$((counter+1))
    percent=$((100*counter/filecount))
    printf "\rProgress: [%-50s] %d%%, processed %d out of %d files." $(printf '%.0s#' $(seq 1 $((percent/2)))) $percent $counter $filecount
done
echo ""