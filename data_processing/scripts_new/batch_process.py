from pathlib import Path
import numpy as np
import csv
import time
import gc
import os
import psutil  # For memory tracking
import sys

def process_in_batches(input_root, batch_size=100, start_batch=0):
    """
    Process files in smaller batches with restarts between batches.
    
    Args:
        input_root (Path): Path to data directory
        batch_size (int): Number of files to process in each batch
        start_batch (int): Which batch to start from (0-based)
    """
    input_root = Path(input_root)
    
    print(f"Starting batch processing with {batch_size} files per batch")
    
    # Initialize stats storage - this will persist across batches
    stats_file = input_root / "batch_stats.npz"
    if start_batch == 0 or not stats_file.exists():
        # Initialize new stats
        global_sum = np.zeros(4, dtype=np.float64)
        global_sumsq = np.zeros(4, dtype=np.float64)
        global_min = np.full(4, np.inf, dtype=np.float64)
        global_max = np.full(4, -np.inf, dtype=np.float64)
        global_count = 0
        processed_ids = {}  # Dictionary to store IDs for each split
    else:
        # Load previous stats
        print(f"Loading previous stats from {stats_file}")
        stats = np.load(stats_file)
        global_sum = stats['sum']
        global_sumsq = stats['sumsq']
        global_min = stats['min']
        global_max = stats['max']
        global_count = stats['count'].item()  # Convert from 0d array to scalar
        processed_ids = dict(stats['processed_ids'].item())
    
    for split in ['train', 'test', 'val']:
        if split not in processed_ids:
            processed_ids[split] = []
        
        before_path = input_root / split / 'before'
        if not before_path.exists():
            print(f"Warning: {before_path} does not exist. Skipping {split} split.")
            continue
            
        # Get all files for this split
        all_files = sorted(list(before_path.glob('*.npy')))
        total_files = len(all_files)
        
        if total_files == 0:
            print(f"Warning: No .npy files found in {before_path}. Skipping {split} split.")
            continue
            
        print(f"Found {total_files} files in {split}/before/")
        
        # Calculate batch information
        total_batches = (total_files + batch_size - 1) // batch_size  # Ceiling division
        
        # Process each batch
        for batch_num in range(start_batch, total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, total_files)
            
            print(f"\n{'='*40}")
            print(f"Processing {split} batch {batch_num+1}/{total_batches} (files {batch_start+1}-{batch_end})")
            print(f"{'='*40}")
            
            # Track memory usage
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Memory usage before batch: {mem_before:.2f} MB")
            
            batch_start_time = time.time()
            
            # Process files in this batch
            for idx in range(batch_start, batch_end):
                file_idx = idx + 1  # 1-based index for display
                before_file = all_files[idx]
                
                print(f"Processing file #{file_idx}/{total_files}: {before_file.name}")
                
                try:
                    # Generate image ID
                    image_id = f"{file_idx:05d}"
                    
                    # Skip if already processed (for resuming)
                    if image_id in processed_ids[split]:
                        print(f"  File #{file_idx} already processed, skipping")
                        continue
                    
                    # Load file
                    print(f"  Loading file #{file_idx}...")
                    file_start_time = time.time()
                    
                    before = np.load(before_file)
                    
                    # Check shape
                    if before.shape[0] != 4:
                        print(f"  Warning: Expected 4 bands but got {before.shape}. Skipping.")
                        continue
                    
                    # Compute stats for train split
                    if split == 'train':
                        print(f"  Computing statistics for file #{file_idx}...")
                        
                        for b in range(4):
                            band_data = before[b, :, :]
                            global_sum[b] += np.sum(band_data, dtype=np.float64)
                            global_sumsq[b] += np.sum(np.square(band_data), dtype=np.float64)
                            global_min[b] = min(global_min[b], np.min(band_data))
                            global_max[b] = max(global_max[b], np.max(band_data))
                        
                        # Update pixel count
                        global_count += before.shape[1] * before.shape[2]
                    
                    # Add to processed IDs
                    processed_ids[split].append(image_id)
                    
                    # Cleanup
                    del before
                    
                    if idx % 10 == 0:
                        gc.collect()
                    
                    file_time = time.time() - file_start_time
                    print(f"  Completed file #{file_idx} in {file_time:.2f} seconds")
                    
                except Exception as e:
                    print(f"  Error processing file #{file_idx}: {e}")
                    continue
            
            # End of batch - save progress
            batch_time = time.time() - batch_start_time
            print(f"\nBatch {batch_num+1} completed in {batch_time:.2f} seconds")
            
            # Save current stats to file
            print(f"Saving statistics after batch {batch_num+1}...")
            np.savez(
                stats_file,
                sum=global_sum,
                sumsq=global_sumsq,
                min=global_min,
                max=global_max,
                count=np.array(global_count),
                processed_ids=np.array(processed_ids, dtype=object)
            )
            
            # Write current IDs to text files
            for s in processed_ids:
                txt_path = input_root / f"{s}.txt"
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(processed_ids[s]))
                print(f"Saved {len(processed_ids[s])} IDs to {txt_path}")
            
            # Report memory usage
            gc.collect()  # Force garbage collection
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Memory usage after batch: {mem_after:.2f} MB (change: {mem_after-mem_before:.2f} MB)")
            
            # Optionally, you could restart the kernel here if running in a notebook
            # But since we can't do that programmatically, we'll just save progress
    
    # All batches completed - calculate final statistics
    if global_count > 0:
        print("\nComputing final statistics...")
        
        mean = global_sum / global_count
        variance = global_sumsq / global_count - np.square(mean)
        variance = np.maximum(variance, 0)  # Handle numerical issues
        std = np.sqrt(variance)
        
        band_labels = ["band2", "band3", "band4", "band8"]
        
        csv_path = input_root / "train_aggregated_band_statistics.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['band', 'mean', 'std', 'min', 'max']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, label in enumerate(band_labels):
                writer.writerow({
                    'band': label,
                    'mean': f"{mean[i]:.6f}",
                    'std':  f"{std[i]:.6f}",
                    'min':  f"{global_min[i]:.6f}",
                    'max':  f"{global_max[i]:.6f}"
                })
        
        print(f"Final statistics saved to {csv_path}")
        print("Band statistics:")
        for i, label in enumerate(band_labels):
            print(f"  {label}: mean={mean[i]:.6f}, std={std[i]:.6f}, min={global_min[i]:.6f}, max={global_max[i]:.6f}")
    
    print("\nBatch processing complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_process.py /path/to/data [batch_size] [start_batch]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    batch_size = 50  # Default
    start_batch = 0  # Default
    
    if len(sys.argv) > 2:
        batch_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        start_batch = int(sys.argv[3])
    
    print(f"Processing data in {input_path} with batch size {batch_size}, starting at batch {start_batch}")
    process_in_batches(input_path, batch_size, start_batch)