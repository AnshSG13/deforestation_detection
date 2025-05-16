import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import random
import multiprocessing

import data_config

def normalize_to_unit_range(tensor):
    """
    Normalize each channel in the tensor to the [0, 1] range.
    For each channel, subtract the channel minimum and divide by the channel range.
    """
    norm_tensor = tensor.clone()
    for c in range(tensor.size(0)):
        c_min = tensor[c].min()
        c_max = tensor[c].max()
        if c_max - c_min > 0:
            norm_tensor[c] = (tensor[c] - c_min) / (c_max - c_min)
        else:
            norm_tensor[c] = torch.zeros_like(tensor[c])
    return norm_tensor

def npy_to_tensor(npy_array):
    """
    Convert a numpy array to a tensor without additional transforms.
    Handles both channels-last (H, W, C) and channels-first (C, H, W) formats.
    
    For a 7-band image, selects bands 0, 1, and 3 directly so that the output
    tensor shape is (3, H, W). For grayscale images, the output tensor shape is (1, H, W).
    """
    npy_array = npy_array.astype(np.float32)
    # For grayscale images (e.g., shape: (H, W))
    if npy_array.ndim == 2:
        return torch.from_numpy(npy_array).unsqueeze(0).float()
    
    # For 3D arrays, check whether channels are the first or last dimension.
    if npy_array.ndim == 3:
        # Case 1: Channels last (H, W, C)
        if npy_array.shape[-1] in [3, 4, 7]:
            if npy_array.shape[-1] == 7:
                # Directly select bands 0, 1, and 3 from the 7-band image
                npy_array = npy_array[:, :, [0, 1, 3]]
            elif npy_array.shape[-1] == 4:
                # Select bands 0, 1, and 3 as before
                npy_array = npy_array[:, :, [0, 1, 3]]
            # Permute to (C, H, W)
            tensor = torch.from_numpy(npy_array).permute(2, 0, 1).float()
            
        # Case 2: Channels first (C, H, W)
        elif npy_array.shape[0] in [3, 4, 7]:
            if npy_array.shape[0] == 7:
                # Directly select bands 0, 1, and 3 from the 7-band image
                npy_array = npy_array[[0, 1, 3], :, :]
            elif npy_array.shape[0] == 4:
                # Select bands 0, 1, and 3 as before
                npy_array = npy_array[[0, 1, 3], :, :]
            tensor = torch.from_numpy(npy_array).float()
            
        else:
            # Default: assume channels last.
            tensor = torch.from_numpy(npy_array).permute(2, 0, 1).float()
            
        return tensor

def load_chips_list(root_dir, split='train'):
    """
    Return a list of chip IDs (strings) for the given split (train, val, or test).
    Assumes the presence of a file named '{split}.txt' listing the chip IDs.
    """
    list_file = os.path.join(root_dir, f"{split}.txt")
    with open(list_file, 'r') as f:
        chip_ids = [line.strip() for line in f if line.strip()]
    return chip_ids

class CDDataset(Dataset):
    """
    A PyTorch Dataset that loads satellite 'before' and 'after' data and a mask.
    Optionally applies dynamic random rotations each time __getitem__ is called,
    when 'augment=True' and the mask has deforestation (value 1).
    """
    def __init__(self, root_dir, split='train', augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.augment = augment
        
        # The subfolder structure is assumed to be:
        # root_dir/
        #    train/ or val/ or test/
        #       before/
        #       after/
        #       mask/
        #    train.txt, val.txt, test.txt
        #
        # We load just the chip IDs here.
        self.chip_ids = load_chips_list(root_dir, split=split)
        
        # Store folder references for easier path building.
        self.split_folder = os.path.join(root_dir, split)
        self.before_folder = os.path.join(self.split_folder, 'before')
        self.after_folder  = os.path.join(self.split_folder, 'after')
        self.mask_folder   = os.path.join(self.split_folder, 'mask')
    
    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, index):
        # Get chip_id, ensuring it has proper naming convention, e.g. chip_XXX.npz
        chip_id = self.chip_ids[index]
        if not chip_id.startswith("chip_"):
            chip_id = f"chip_{chip_id}"
        if not chip_id.endswith(".npz"):
            chip_id = f"{chip_id}.npz"
        
        # Build paths
        before_path = os.path.join(self.before_folder, chip_id)
        after_path  = os.path.join(self.after_folder, chip_id)
        mask_path   = os.path.join(self.mask_folder, chip_id)
        
        # Load .npz arrays
        before_arr = np.load(before_path)['array'] if before_path.endswith('.npz') else np.load(before_path)
        after_arr  = np.load(after_path)['array']  if after_path.endswith('.npz')  else np.load(after_path)
        mask_arr   = np.load(mask_path)['array']   if mask_path.endswith('.npz')   else np.load(mask_path)
        
        # Convert numpy arrays to tensors
        before_tensor = npy_to_tensor(before_arr)
        after_tensor  = npy_to_tensor(after_arr)
        mask_tensor   = npy_to_tensor(mask_arr).long()  # For masks, keep as integer type
        
        # Normalize 'before' and 'after' images
        before_tensor = normalize_to_unit_range(before_tensor)
        after_tensor  = normalize_to_unit_range(after_tensor)

        # Optionally augment via random rotation if:
        #   1) self.augment == True
        #   2) the mask contains value 1 (deforestation)
        if self.augment and (mask_tensor == 1).any():
            # Randomly choose among 0, 1, 2, or 3 rotations (0°, 90°, 180°, 270°)
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                before_tensor = torch.rot90(before_tensor, k=k, dims=[1, 2])
                after_tensor  = torch.rot90(after_tensor,  k=k, dims=[1, 2])
                mask_tensor   = torch.rot90(mask_tensor,   k=k, dims=[1, 2])
        
        sample = {
            'A': before_tensor,
            'B': after_tensor,
            'L': mask_tensor,
            'list': chip_id  # or you can store chip_id without .npz
        }
        return sample

# ── balanced dataset ────────────────────────────────────────────────────────────

class BalancedCDDataset(CDDataset):
    """
    Like CDDataset but returns an *equal* number of deforested and
    non‑deforested chips every epoch.

    A chip is *positive* if ≥ `thr` fraction of its pixels are 1 in the mask.
    """
    def __init__(self, root_dir, split='train', augment=False, thr: float = 0.00):
        super().__init__(root_dir=root_dir, split=split, augment=augment)
        self.thr = thr
        # build positive / negative index lists
        self.pos_idx, self.neg_idx = self._build_index_lists()
        if not self.pos_idx:
            raise RuntimeError(f"No chips with ≥{thr*100:.1f}% deforestation!")
        # for reproducible val splits
        self.rng = random.Random(0 if split == 'val' else None)

    def _build_index_lists(self):
        pos, neg = [], []
        for i, chip_id in enumerate(self.chip_ids):
            # ensure file name has chip_ prefix and .npz suffix
            fid = chip_id
            if not fid.startswith("chip_"):
                fid = f"chip_{fid}"
            if not fid.endswith(".npz"):
                fid = f"{fid}.npz"
            mask_path = os.path.join(self.mask_folder, fid)
            arr = np.load(mask_path)['array'] if mask_path.endswith('.npz') else np.load(mask_path)
            ratio = (arr == 1).mean()
            (pos if ratio >= self.thr else neg).append(i)
        return pos, neg

    def __len__(self):
        # show DataLoader 2× #positive so we can balance
        return 2 * len(self.pos_idx)

    def __getitem__(self, index):
        # even → positive, odd → random negative
        if index % 2 == 0:
            real_idx = self.pos_idx[index // 2]
        else:
            real_idx = self.rng.choice(self.neg_idx)
        return super().__getitem__(real_idx)


# ── loader factory ──────────────────────────────────────────────────────────────

def get_loaders(args):
    """
    Returns DataLoaders for train/val/test splits.

      - train/val: BalancedCDDataset (equal positive/negative at threshold)
      - test     : full CDDataset
    """
    data_cfg = data_config.DataConfig().get_data_config(args.data_name)
    root_dir = data_cfg.root_dir
    thr = getattr(args, 'deforestation_thr', 0.0000000001)

    # train
    train_ds = BalancedCDDataset(
        root_dir=root_dir,
        split='train',
        augment=True,
        thr=thr
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # val
    val_ds = BalancedCDDataset(
        root_dir=root_dir,
        split='val',
        augment=False,
        thr=thr
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # test (full dataset)
    test_ds = CDDataset(
        root_dir=root_dir,
        split='test',
        augment=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    return {
        'train': train_loader,
        'val':   val_loader,
        'test':  test_loader
    }


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis

def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5

def get_device(args):
    # Set GPU IDs from args.gpu_ids (expects a comma-separated string).
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
