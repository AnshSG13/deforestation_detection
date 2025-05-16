import rasterio
from rasterio.enums import Resampling
import yaml

def load_config(config_path='src/config/config.yaml'):
    """
    Loads the YAML configuration file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_bounds(src):
    transform = src.transform
    width = src.width
    height = src.height
    minx, maxy = transform * (0, 0)
    maxx, miny = transform * (width, height)
    bounds = {
        'left': minx,
        'bottom': miny,
        'right': maxx,
        'top': maxy
    }
    return bounds

def resample_raster(src_path, dst_path, scale_factor, resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        data = src.read(
            out_shape=(
                src.count,
                int(src.height * scale_factor),
                int(src.width * scale_factor)
            ),
            resampling=resampling
        )
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )

        profile = src.profile
        profile.update({
            'height': data.shape[1],
            'width': data.shape[2],
            'transform': transform
        })

        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(data)

def create_chips(image, chip_size=(256, 256), stride=256):
    """
    Splits the image into smaller chips.

    Parameters:
    - image: Numpy array of the image with shape (bands, height, width).
    - chip_size: Tuple (height, width) of the chips.
    - stride: Stride between chips.

    Returns:
    - chips: List of image chips as numpy arrays.
    """
    bands, height, width = image.shape
    chip_height, chip_width = chip_size
    chips = []

    # Iterate over the image dimensions
    for y in range(0, height - chip_height + 1, stride):
        for x in range(0, width - chip_width + 1, stride):
            chip = image[:, y:y+chip_height, x:x+chip_width]
            chips.append(chip)

    return chips


def augment_chip(chip, label=None):
    """
    Performs augmentation on the chip and label, including rotations and flips.

    Parameters:
    - chip: Numpy array of the image chip.
    - label: Numpy array of the label mask (if any).

    Returns:
    - augmented_chips: List of augmented chips.
    - augmented_labels: List of augmented labels (if labels are provided).
    """
    import numpy as np

    augmented_chips = [chip]
    augmented_labels = [label] if label is not None else None

    # Define augmentation operations
    operations = [
        lambda x: np.rot90(x, k=1, axes=(1, 2)),   # Rotate 90 degrees
        lambda x: np.rot90(x, k=2, axes=(1, 2)),   # Rotate 180 degrees
        lambda x: np.flip(x, axis=1),              # Flip vertically
        lambda x: np.flip(x, axis=2),              # Flip horizontally
    ]

    for op in operations:
        augmented_chip = op(chip)
        augmented_chips.append(augmented_chip)
        if label is not None:
            # Apply the same operation to the label
            if label.ndim == 2:
                label_expanded = label[np.newaxis, ...]  # Add band dimension
                augmented_label = op(label_expanded)[0]
            else:
                augmented_label = op(label)
            augmented_labels.append(augmented_label)

    return augmented_chips, augmented_labels

