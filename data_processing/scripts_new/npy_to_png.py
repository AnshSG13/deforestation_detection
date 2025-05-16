from pathlib import Path
import numpy as np
from PIL import Image
import csv

def convert_chips_to_rgb(input_root, output_root):
    """
    Convert .npy files to RGB PNGs maintaining the train/test/val structure.
    Additionally, for the train split, compute aggregated statistics (min, max, std, and mean)
    for the last 3 bands (band3, band4, band8) across all chips in the 'before' folder.
    
    Args:
        input_root: Path to the input directory containing train/test/val folders.
        output_root: Path to output directory where RGB PNGs and statistics CSV will be saved.
    """
    # Create output directory structure
    output_root.mkdir(parents=True, exist_ok=True)
    
    def process_image(img):
        # Take last 3 bands and transpose to (H, W, C) format for PIL.
        # Also, normalize each channel to the 0-255 range.
        img_subset = img[-3:, :, :]
        norm_img = np.empty_like(img_subset, dtype=np.float32)
        for i in range(norm_img.shape[0]):
            band = img_subset[i]
            norm_img[i] = (band - band.min()) / (band.max() - band.min()) * 255
        # Transpose to (H, W, C).
        return norm_img.transpose(1, 2, 0).astype(np.uint8)
    
    # Initialize global statistics for the training split over the 3 bands.
    # We will accumulate sum, sumsq, min, and max for each band.
    global_sum = None
    global_sumsq = None
    global_min = None
    global_max = None
    global_count = 0  # Number of pixels per band across all chips
    
    # Process each split: train, test, and val.
    for split in ['train', 'test', 'val']:
        print(f"Processing {split} split...")
        
        # Create output directories for this split.
        split_dir = output_root / split
        (split_dir / 'T1').mkdir(parents=True, exist_ok=True)
        (split_dir / 'T2').mkdir(parents=True, exist_ok=True)
        (split_dir / 'GT').mkdir(parents=True, exist_ok=True)
        
        # Get all files in the 'before' folder.
        before_files = sorted(list((input_root / split / 'before').glob('*.npy')))
        
        processed_ids = []
        for idx, before_file in enumerate(before_files, 1):
            try:
                chip_id = before_file.stem
                image_id = f'{idx:05d}'  # Format: 00001, 00002, etc.
                
                # Load corresponding files.
                before = np.load(input_root / split / 'before' / f'{chip_id}.npy')
                after  = np.load(input_root / split / 'after' / f'{chip_id}.npy')
                mask   = np.load(input_root / split / 'mask' / f'{chip_id}.npy')
                
                # For train split, update global statistics over the last 3 bands.
                if split == 'train':
                    # Extract the last 3 bands corresponding to band3, band4, and band8.
                    bands = before[-3:, :, :]
                    if global_sum is None:
                        num_bands = bands.shape[0]  # should be 3
                        global_sum = np.zeros(num_bands, dtype=np.float64)
                        global_sumsq = np.zeros(num_bands, dtype=np.float64)
                        global_min = np.full(num_bands, np.inf)
                        global_max = np.full(num_bands, -np.inf)
                    
                    # For each band update the running sums, squared sums, min and max.
                    for i in range(bands.shape[0]):
                        band_data = bands[i, :, :]
                        global_sum[i] += band_data.sum()
                        global_sumsq[i] += np.square(band_data).sum()
                        global_min[i] = min(global_min[i], band_data.min())
                        global_max[i] = max(global_max[i], band_data.max())
                    # Update total count of pixels per band.
                    global_count += bands.shape[1] * bands.shape[2]
                
                # Process and save before image (T1).
                before_processed = process_image(before)
                Image.fromarray(before_processed, 'RGB').save(split_dir / 'T1' / f'{image_id}.png')
                
                # Process and save after image (T2).
                after_processed = process_image(after)
                Image.fromarray(after_processed, 'RGB').save(split_dir / 'T2' / f'{image_id}.png')
                
                # Process and save mask (GT). Scale mask values to [0, 255].
                mask_processed = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_processed).save(split_dir / 'GT' / f'{image_id}.png')
                
                processed_ids.append(image_id)
                
                if idx % 100 == 0:
                    print(f"Processed {idx} files in {split}")
                
            except Exception as e:
                print(f"Error processing chip {chip_id} in {split}: {str(e)}")
                continue
        
        # Create a text file listing processed image IDs for this split.
        with open(output_root / f'{split}.txt', 'w') as f:
            f.write('\n'.join(processed_ids))
        
        print(f"Completed {split} split: {len(processed_ids)} files processed")
    
    # After processing the training split, compute aggregated statistics per band.
    if global_sum is not None and global_count > 0:
        mean = global_sum / global_count
        variance = global_sumsq / global_count - np.square(mean)
        std = np.sqrt(variance)
        
        # Create CSV with a single row per band.
        band_labels = ["band3", "band4", "band8"]
        csv_path = output_root / 'train_aggregated_band_statistics.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['band', 'mean', 'std', 'min', 'max']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, label in enumerate(band_labels):
                writer.writerow({
                    'band': label,
                    'mean': f"{mean[i]:.4f}",
                    'std': f"{std[i]:.4f}",
                    'min': f"{global_min[i]:.4f}",
                    'max': f"{global_max[i]:.4f}"
                })
        print(f"Aggregated training band statistics saved to {csv_path}")

# Example usage:
# input_root = Path("/path/to/input/chips")   # This folder should contain train, test, and val subfolders.
# output_root = Path("/path/to/output/pngs")
# convert_chips_to_rgb(input_root, output_root)
