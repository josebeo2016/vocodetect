#!/bin/bash

##################
# 

source_dir=$1    # Replace with the source directory path
target_dir=$2    # Replace with the target directory path

cd "$source_dir"

for file in *; do
    if [[ -f $file ]]; then
        dir_name=$(basename "$source_dir")
        new_name=$(echo "$dir_name"_"$file" | tr '/' '_')
        mv "$file" "$target_dir/$new_name"
        echo "Moved file '$file' to '$target_dir/$new_name'"
    fi
done