# pipeline.py
from pathlib import Path
import numpy as np
import rasterio
import argparse

from npy_txt import compute_stats_and_create_txt_band2_3_4_8_single_folder
from npy_to_png import convert_chips_to_rgb

def parse_args():
    parser = argparse.ArgumentParser(description='npy to tiff or png conversion')
    parser.add_argument('--output-select', type=str, required=True,
                        help='download as png and/or compute stats')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Base output directory for processed chips')
    return parser.parse_args()

def run_pipeline(args):
    output_dir = Path(args.output_dir)
    input_root = output_dir / "chips" 
    output_root = output_dir / "png_chips"
    
    # Use the correct attribute name: args.output_select
    if args.output_select == "txt":
        compute_stats_and_create_txt_band2_3_4_8_single_folder(input_root)
    else:
        convert_chips_to_rgb(input_root, output_root)
        
    print("Pipeline complete.")

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
