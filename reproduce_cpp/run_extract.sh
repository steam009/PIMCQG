#!/bin/bash

# Use wildcards to match all qualifying files
for input_file in nohup_dpu*_nprobe*_gist.out; do
    # Check if files exist (avoid issues when wildcard matches nothing)
    if [ ! -f "$input_file" ]; then
        continue
    fi
    # Replace _naive.out in input filename with _extracted.csv to generate output filename
    output_file="${input_file/.out/_extracted.csv}"
    python extract_data.py "$input_file" "$output_file"
done