#!/bin/bash

echo "## 100 spoof from LA19 train and dev set"
while IFS= read -r line
do
  filename=$(echo "$line" | awk '{print $2}') # Change to the correct column if $2 is not correct.
  input="/dataa/phucdt/vocodetect/traindata/la19_lyra/la19_eval_lyra/${filename}_decoded.wav" # Replace with proper path
  output="./la19_spoof_100/" # Replace with proper path
  cp "$input" "$output"
done < <(grep spoof /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt | shuf -n 100)


echo "## 100 spoof from LA19 train and dev set"
while IFS= read -r line
do
  filename=$(echo "$line" | awk '{print $2}') # Change to the correct column if $2 is not correct.
  input="/dataa/phucdt/vocodetect/traindata/la19_lyra/la19_eval_lyra/${filename}_decoded.wav" # Replace with proper path
  output="./la19_bona_100/" # Replace with proper path
  cp "$input" "$output"
done < <(grep bonafide /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt | shuf -n 100)