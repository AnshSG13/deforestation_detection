import os
import numpy as np
import dask.array as da
from dask import delayed
from pathlib import Path
import argparse
from dask import compute
import dask

# Utility to save chip with metadata
def _save_chip_with_position(idx, b, a, m, split, y_start, x_start, chip_size, index_file, output_dir):
    prefix = f'chip_{idx:05d}'
    for name, arr in [('before', b), ('after', a), ('mask', m)]:
        path = Path(output_dir, split, name)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / f'{prefix}.npz', array=arr)
    index_file.write(f"{prefix},{split},{y_start},{x_start},{chip_size}\n")

# Sliding window with edge flush
def sliding_window(arr, win, step):
    h, w = win
    H, W = arr.shape[-2], arr.shape[-1]
    y_starts = list(range(0, H - h + 1, step))
    if y_starts[-1] != H - h:
        y_starts.append(H - h)
    x_starts = list(range(0, W - w + 1, step))
    if x_starts[-1] != W - w:
        x_starts.append(W - w)
    windows = da.stack([arr[..., ys:ys+h, xs:xs+w] for ys in y_starts for xs in x_starts])
    return windows, len(y_starts), len(x_starts)

# Core function
def chunked_create_and_split_image_chips(
    before_path, after_path, mask_path,
    chip_size=256, overlap=10, chunk_core=2048,
    output_dir='chips', train_start_index=0,
    test_start_index=0, val_start_index=0, split_probs=None):
    """
    Create chips and split into train/val/test.
    For first two locations: first 90% of windows are train, last 10% are val.
    For third (test-only) location: all windows are test.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    idx_file = open(Path(output_dir)/'chip_positions.csv','w')
    idx_file.write("chip_id,split,y_start,x_start,chip_size\n")

    # load arrays
    before_full = np.load(before_path)
    after_full  = np.load(after_path)
    mask_full   = np.load(mask_path)
    _, H, W = before_full.shape
    idx_file.write(f"# original_full_array_size,{H},{W}\n")

    # default split
    if split_probs is None:
        split_probs = [0.9, 0.0, 0.1]
    train_frac, test_frac, val_frac = split_probs

    step = chip_size - overlap
    chunk_size = chunk_core + step
    rng = np.random.default_rng(0)
    train_i, test_i, val_i = train_start_index, test_start_index, val_start_index

    y=0
    while y < H:
        y_end = min(y+chunk_size, H)
        if y_end-y < chip_size: break
        x=0
        while x < W:
            x_end = min(x+chunk_size, W)
            if x_end-x < chip_size: break

            before = da.from_array(before_full[:,y:y_end,x:x_end], chunks=('auto',)*3)
            after  = da.from_array(after_full[:,y:y_end,x:x_end], chunks=('auto',)*3)
            mask   = da.from_array(mask_full[y:y_end,x:x_end], chunks=('auto',)*2)
            both_valid = ((before[-1]==0)&(after[-1]==0)).astype(np.uint8)
            before = da.concatenate([before, both_valid[None]], axis=0)
            after  = da.concatenate([after,  both_valid[None]], axis=0)

            before_w, n_y, n_x = sliding_window(before, (chip_size,chip_size), step)
            after_w,  _,   _   = sliding_window(after,  (chip_size,chip_size), step)
            mask_w,   _,   _   = sliding_window(mask,   (chip_size,chip_size), step)

            n_chips = n_y * n_x
            b_arr = before_w.reshape(n_chips, -1, chip_size, chip_size).compute()
            a_arr = after_w.reshape(n_chips, -1, chip_size, chip_size).compute()
            m_arr = mask_w.reshape(n_chips, chip_size, chip_size).compute()

            # compute cutoff for deterministic split when no test_frac
            if test_frac == 0:
                # two-way: train first chunk, val remainder
                cut = int(n_chips * train_frac / (train_frac + val_frac))
            tasks=[]
            for chip_idx in range(n_chips):
                # compute start positions
                y_idx = chip_idx // n_x
                x_idx = chip_idx %  n_x
                y_start = y + y_idx*step
                x_start = x + x_idx*step
                # assign split
                if test_frac==0:
                    if chip_idx < cut:
                        split='train'; idx=train_i; train_i+=1
                    else:
                        split='val';   idx=val_i;   val_i+=1
                else:
                    # all test-only location or three-way
                    r = rng.random()
                    if r < train_frac:
                        split='train'; idx=train_i; train_i+=1
                    elif r < train_frac+val_frac:
                        split='val';   idx=val_i;   val_i+=1
                    else:
                        split='test';  idx=test_i;  test_i+=1
                tasks.append(delayed(_save_chip_with_position)(
                    idx, b_arr[chip_idx], a_arr[chip_idx], m_arr[chip_idx],
                    split, y_start, x_start, chip_size, idx_file, output_dir
                ))
            dask.compute(*tasks)
            x += chunk_core
        y += chunk_core

    total = (train_i-train_start_index)+(test_i-test_start_index)+(val_i-val_start_index)
    print(f"Finished. Train {train_i} Test {test_i} Val {val_i} Total {total}")
    idx_file.close()
    return train_i, test_i, val_i




def process_chips_strategic_split(input_dirs, output_base, chip_size=256, overlap=0, chunk_core=2048, min_deforestation_pixels=0):
    train_index = test_index = val_index = 0

    for i, input_dir in enumerate(input_dirs):
        input_base = Path(input_dir)
        before_path = input_base / 'sentinel2/combined/Sentinel2_july2022.npy'
        after_path  = input_base / 'sentinel2/combined/Sentinel2_july2023.npy'
        mask_path   = input_base / 'masks/prodes_mask_aug2022_jun2023.npy'
        output_dir  = Path(output_base) / 'chips'
        # log the original full-array size per location
        arr = np.load(before_path)
        _, orig_H, orig_W = arr.shape
        print(f"Location {i+1}: original size (H, W) = ({orig_H}, {orig_W})")

        # decide per–location behavior
        if   i < 2:
            # first two dirs → only train & val, keep your overlap
            split_probs = [0.85, 0.0, 0.15]    # [train, test, val]
            this_overlap = overlap
        elif i == 2:
            # third dir → 100% test, no overlap
            split_probs = [0.0, 1.0, 0.0]
            this_overlap = 0
        else:
            # any extra dirs? funnel into train
            split_probs = [1.0, 0.0, 0.0]
            this_overlap = overlap

        train_index, test_index, val_index = chunked_create_and_split_image_chips(
            before_path=str(before_path),
            after_path=str(after_path),
            mask_path=str(mask_path),
            chip_size=chip_size,
            overlap=this_overlap,
            chunk_core=chunk_core,
            output_dir=output_dir,
            split_probs=split_probs, 
            train_start_index=train_index,
            test_start_index=test_index,
            val_start_index=val_index,
        )
        print(f"Updated counters: Train={train_index}, Test={test_index}, Val={val_index}")

    total = train_index + test_index + val_index
    print("\nAll directories processed.")
    print(f"Total Train Chips: {train_index}")
    print(f"Total Test Chips:  {test_index}")
    print(f"Total Val Chips:   {val_index}")
    print(f"Grand Total:       {total}")



def main():
    parser = argparse.ArgumentParser(description="Chunked chip extraction with strategic train/val/test splitting.")
    parser.add_argument('--input_dirs', nargs='+', required=True, help="One or more directories with data.")
    parser.add_argument('--output_dir', required=True, help="Output directory for chips.")
    parser.add_argument('--chip_size', type=int, default=256, help="Chip size (e.g. 128)")
    parser.add_argument('--overlap', type=int, default=10, help="Overlap in pixels (e.g. 90 => ~70%% overlap)")
    parser.add_argument('--chunk_core', type=int, default=2048,
                        help="Core region for each chunk; chunk size = chunk_core + (chip_size - overlap)")
    parser.add_argument('--min_deforestation', type=int, default=0,
                        help="Minimum deforestation pixels required in a chip.")
    args = parser.parse_args()

    process_chips_strategic_split(
        input_dirs=args.input_dirs,
        output_base=Path(args.output_dir),
        chip_size=args.chip_size,
        overlap=args.overlap,
        chunk_core=args.chunk_core,
        min_deforestation_pixels=args.min_deforestation
    )


if __name__ == '__main__':
    main()
