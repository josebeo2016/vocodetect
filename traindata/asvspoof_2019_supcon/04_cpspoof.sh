#!/bin/bash

# echo "## spoof from LA19 train"
# while IFS= read -r line
# do
#   filename=$(echo "$line" | awk '{print $2}') # Change to the correct column if $2 is not correct.
#   input="/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_train/wav/$filename.wav" # Replace with proper path
#   output="./spoof_train/" # Replace with proper path
#   cp "$input" "$output"
# done < <(grep spoof /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt)

echo "## spoof from LA19 dev set"
while IFS= read -r line
do
  filename=$(echo "$line" | awk '{print $2}') # Change to the correct column if $2 is not correct.
  input="/datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_dev/wav/$filename.wav.wav" # Replace with proper path
  output="./spoof_dev/$filename.wav" # Replace with proper path
  cp "$input" "$output"
done < <(grep spoof /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt)