#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract all pattern-matching data blocks from nohup1.out and save to CSV file
"""

import re
import csv
from pathlib import Path

def extract_all_data_blocks(input_file, output_csv):
    """
    Extract all pattern-matching data blocks from nohup file and save to CSV
    
    Args:
        input_file: path to input nohup file
        output_csv: path to output CSV file
    """
    # Read entire file
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Define regex pattern for data blocks
    # Support two formats:
    # 1. Full format: includes "Average workload per DPU" and "2560DPU:EF"
    # 2. Simplified format: only up to "EF: XX, POST EF: XX, NPROBE: XX, QPS: XX, Recall: XX%"
    
    # Common part
    common_part = (
        r'Query prepare time:\s*([\d.]+)\s*seconds\s*'
        r'(?:max_query_cluster_per_dpu:\s*\d+\s*\n)?'  # optional max_query_cluster_per_dpu line
        r'CPU-DPU data transfer time:\s*([\d.]+)\s*seconds\s*'
        r'Script is running\. PID:\s*(\d+)\s*'
        r'Attach perf stat now, then press Enter to continue\.\.\.\s*'
        r'DPU search time:\s*([\d.]+)\s*seconds\s*'
        r'DPU-CPU data transfer time:\s*([\d.]+)\s*seconds\s*'
        r'CPU Post process time:\s*([\d.]+)\s*seconds\s*'
        r'EF:\s*(\d+),\s*POST EF:\s*(\d+),\s*NPROBE:\s*(\d+),\s*QPS:\s*([\d.]+),\s*Recall:\s*([\d.]+)%\s*'
    )
    
    # Full format follow-up part (optional)
    full_format_part = (
        r'Average workload per DPU:\s*(\d+)\s*'
        r'(?:.*?\n)*?'  # match possible blank lines and other content in between
        r'=== DPU Load Balancing Results ===\s*'
        r'Total time:\s*([\d.]+)\s*seconds\s*'
        r'CPU version search completed\s*'
        r'2560DPU:EF:\s*(\d+),\s*POST EF:\s*(\d+),\s*NPROBE:\s*(\d+),\s*QPS:\s*([\d.]+),\s*Recall:\s*([\d.]+)%'
    )
    
    # Combined pattern: match full format or simplified format
    block_pattern = re.compile(
        common_part + 
        r'(?:' + full_format_part + r')?',  # full format part is optional
        re.MULTILINE | re.DOTALL
    )
    
    # Find all matching data blocks
    matches = block_pattern.findall(content)
    
    if not matches:
        print("No matching data blocks found")
        return
    
    # Prepare CSV data
    csv_data = []
    
    # Define CSV column order
    csv_columns = [
        'Query Prepare Time (seconds)',
        'CPU-DPU Transfer Time (seconds)',
        'PID',
        'DPU Search Time (seconds)',
        'DPU-CPU Transfer Time (seconds)',
        'CPU Post Process Time (seconds)',
        'EF',
        'POST EF',
        'NPROBE',
        'QPS',
        'Recall (%)',
        'Average Workload per DPU',
        'Total Time (seconds)',
        'EF (2560DPU)',
        'POST EF (2560DPU)',
        'NPROBE (2560DPU)',
        'QPS (2560DPU)',
        'Recall (2560DPU) (%)',
    ]
    
    # Process each matching data block
    for match in matches:
        # Determine whether full format or simplified format
        # Full format has 18 capture groups; simplified format has only 11
        if len(match) >= 18 and match[11]:  # Full format
            row = {
                'Query Prepare Time (seconds)': match[0],
                'CPU-DPU Transfer Time (seconds)': match[1],
                'PID': match[2],
                'DPU Search Time (seconds)': match[3],
                'DPU-CPU Transfer Time (seconds)': match[4],
                'CPU Post Process Time (seconds)': match[5],
                'EF': match[6],
                'POST EF': match[7],
                'NPROBE': match[8],
                'QPS': match[9],
                'Recall (%)': match[10],
                'Average Workload per DPU': match[11],
                'Total Time (seconds)': match[12],
                'EF (2560DPU)': match[13],
                'POST EF (2560DPU)': match[14],
                'NPROBE (2560DPU)': match[15],
                'QPS (2560DPU)': match[16],
                'Recall (2560DPU) (%)': match[17],
            }
        else:  # Simplified format
            row = {
                'Query Prepare Time (seconds)': match[0],
                'CPU-DPU Transfer Time (seconds)': match[1],
                'PID': match[2],
                'DPU Search Time (seconds)': match[3],
                'DPU-CPU Transfer Time (seconds)': match[4],
                'CPU Post Process Time (seconds)': match[5],
                'EF': match[6],
                'POST EF': match[7],
                'NPROBE': match[8],
                'QPS': match[9],
                'Recall (%)': match[10],
                'Average Workload per DPU': '',  # Simplified format does not have these fields
                'Total Time (seconds)': '',
                'EF (2560DPU)': '',
                'POST EF (2560DPU)': '',
                'NPROBE (2560DPU)': '',
                'QPS (2560DPU)': '',
                'Recall (2560DPU) (%)': '',
            }
        csv_data.append(row)
    
    # Write CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Successfully extracted {len(csv_data)} data blocks")
    print(f"Data saved to: {output_csv}")
    
    # Print preview of first 3 rows
    print("\nFirst 3 rows preview:")
    for i, row in enumerate(csv_data[:3], 1):
        print(f"\nData block {i}:")
        for key, value in row.items():
            if value:
                print(f"  {key}: {value}")

if __name__ == '__main__':
    import sys
    
    # If command line arguments are provided, use the first as input file
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        # Default file path
        input_file = Path(__file__).parent / 'nohup_dpu300_nprobe256.out'
    
    # If a second argument is provided, use it as output file
    if len(sys.argv) > 2:
        output_csv = Path(sys.argv[2])
    else:
        # Default output filename based on input filename
        output_csv = Path(__file__).parent / f'{input_file.stem}_extracted.csv'
    
    # Execute extraction
    extract_all_data_blocks(input_file, output_csv)
