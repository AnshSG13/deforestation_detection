from pathlib import Path
import shutil
import random
import math

def create_subset_dataset(
    full_dataset_path: Path = Path('data/processed/rgb_chips'),
    subset_dataset_path: Path = Path('data/processed/1%_rgb_chips'),
    subset_percentage: float = 0.01,
    random_seed: int = 42
):
    """
    Creates a subset of the dataset with the specified percentage of samples.
    Maintains the same directory structure and updates txt files accordingly.
    
    Args:
        full_dataset_path: Path to the full dataset
        subset_dataset_path: Path where the subset will be saved
        subset_percentage: Percentage of data to include (default: 0.01 for 1%)
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    
    # Create the subset directory structure
    subset_dataset_path.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'test', 'val']:
        if not (full_dataset_path / split).exists():
            continue
            
        print(f"Processing {split} split...")
        
        # Create directories
        for subdir in ['T1', 'T2', 'GT']:
            (subset_dataset_path / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # Read original image IDs from txt file
        try:
            with open(full_dataset_path / f'{split}.txt', 'r') as f:
                original_ids = f.read().splitlines()
        except FileNotFoundError:
            print(f"Warning: {split}.txt not found, skipping {split} split")
            continue
        
        # Calculate number of images for subset
        n_subset = math.ceil(len(original_ids) * subset_percentage)
        print(f"Selecting {n_subset} images from {len(original_ids)} {split} images")
        
        # Randomly select subset of IDs
        selected_ids = random.sample(original_ids, n_subset)
        
        # Copy selected images
        for image_id in selected_ids:
            for subdir in ['T1', 'T2', 'GT']:
                src = full_dataset_path / split / subdir / f'{image_id}.png'
                dst = subset_dataset_path / split / subdir / f'{image_id}.png'
                shutil.copy2(src, dst)
        
        # Create new txt file with selected IDs
        with open(subset_dataset_path / f'{split}.txt', 'w') as f:
            f.write('\n'.join(selected_ids))
        
        print(f"Completed {split} split: {len(selected_ids)} images copied")

if __name__ == "__main__":
    create_subset_dataset()