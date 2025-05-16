from pathlib import Path
import argparse

# Import the necessary modules
import create_chips
from npy_to_png import convert_chips_to_rgb

def parse_args():
    parser = argparse.ArgumentParser(description='Create chips and convert them to RGB images')
    parser.add_argument('--input-dirs', type=str, nargs='+', required=True,
                      help='List of input directories containing processed data')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Base output directory for chips and RGB images')
    parser.add_argument('--chip-size', type=int, default=256,
                      help='Size of output chips (default: 256)')
    parser.add_argument('--overlap', type=int, default=10,
                      help='Overlap between chips (default: 10)')
    parser.add_argument('--min-deforestation', type=int, default=0,
                      help='Minimum deforestation pixels per chip (default: 0)')
    return parser.parse_args()

def run_pipeline(args):
    # Create combined chips from all processed regions
    create_chips.process_chips_strategic_split(
        input_dirs=[Path(dir) for dir in args.input_dirs],
        output_base=Path(args.output_dir),
        chip_size=args.chip_size,
        overlap=args.overlap,
        min_deforestation_pixels=args.min_deforestation
    )

    # Convert chips to RGB images
    # output_dir = Path(args.output_dir)
    # convert_chips_to_rgb(
    #     input_root=output_dir / "chips",
    #     output_root=output_dir / "chips_png"
    # )
    # print("Pipeline complete.")

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)