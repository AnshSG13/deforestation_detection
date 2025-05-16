# pipeline.py
from pathlib import Path
import numpy as np
import rasterio
import argparse

from npy_to_png import convert_chips_to_rgb

def parse_args():
    parser = argparse.ArgumentParser(description='npy to png')
    parser.add_argument('--input-dirs', type=str, nargs='+', required=True,
                      help='List of input directories containing npy chips')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Base output directory for processed chips')
    return parser.parse_args()



def run_pipeline(args):


    output_dir = Path(args.output_dir)
    convert_chips_to_rgb(
        input_root=output_dir / "chips",
        output_root=output_dir / "chips_png"
    )
    print("Pipeline complete.")

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)