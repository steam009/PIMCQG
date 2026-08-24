#!/usr/bin/env python3
"""
Convert pickle mapping files to C++ readable format

Usage:
    python convert_pickle.py input.pkl output.txt
"""

import pickle
import sys
import os
import json
from pathlib import Path

def convert_pickle_to_text(pickle_file, output_file):
    """Convert pickle file to text format"""
    print(f"Reading pickle file: {pickle_file}")
    
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Data size: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        # Write text format
        with open(output_file, 'w') as f:
            if isinstance(data, dict):
                # Dict format: cluster_id -> [original_indices]
                for cluster_id, indices in data.items():
                    f.write(f"{cluster_id}")
                    for idx in indices:
                        f.write(f" {idx}")
                    f.write("\n")
            else:
                print(f"Warning: unsupported data type {type(data)}")
                return False
        
        print(f"Conversion complete: {output_file}")
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

def convert_pickle_to_json(pickle_file, output_file):
    """Convert pickle file to JSON format"""
    print(f"Reading pickle file: {pickle_file}")
    
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to JSON-serializable format
        if isinstance(data, dict):
            json_data = {str(k): v for k, v in data.items()}
        else:
            json_data = data
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Conversion complete: {output_file}")
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

def convert_pickle_to_binary(pickle_file, output_file):
    """Convert pickle file to binary format"""
    print(f"Reading pickle file: {pickle_file}")
    
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        with open(output_file, 'wb') as f:
            if isinstance(data, dict):
                # Write dictionary size
                f.write(len(data).to_bytes(4, 'little'))
                
                for cluster_id, indices in data.items():
                    # Write cluster_id
                    f.write(cluster_id.to_bytes(4, 'little'))
                    # Write indices count
                    f.write(len(indices).to_bytes(4, 'little'))
                    # Write all indices
                    for idx in indices:
                        f.write(idx.to_bytes(4, 'little'))
            else:
                print(f"Warning: unsupported data type {type(data)}")
                return False
        
        print(f"Conversion complete: {output_file}")
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python convert_pickle.py input.pkl output.txt        # Convert to text format")
        print("  python convert_pickle.py input.pkl output.json       # Convert to JSON format")
        print("  python convert_pickle.py input.pkl output.bin        # Convert to binary format")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: input file does not exist: {input_file}")
        sys.exit(1)
    
    # Select conversion format based on output file extension
    output_ext = Path(output_file).suffix.lower()
    
    if output_ext == '.json':
        success = convert_pickle_to_json(input_file, output_file)
    elif output_ext == '.bin':
        success = convert_pickle_to_binary(input_file, output_file)
    else:
        success = convert_pickle_to_text(input_file, output_file)
    
    if success:
        print("Conversion successful!")
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
