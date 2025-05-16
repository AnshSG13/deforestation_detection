import rasterio
from rasterio.enums import Resampling
import subprocess
import numpy as np
from pathlib import Path
import os
import tempfile
from rasterio.errors import RasterioIOError
import re     

print("Loading process_sentinel.py")
#print("Defined functions:", [name for name, obj in globals().items() if callable(obj)])
             

def filter_sentinel_bands(input_dir, output_dir, desired_bands):

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── helper to grab acquisition time from filename ─────────────────────────
    ts_re = re.compile(r"\d{8}T\d{6}")      #   20250414T104601
    def _acq_time(fname: str) -> str:
        m = ts_re.search(fname)
        return m.group(0) if m else "unknown‑time"

    for image_path in input_dir.glob("*.tif"):
        try:
            with rasterio.open(image_path) as src:
                print(f"▶ Processing: {image_path.name}")

                # map from description -> 1‑based band index
                band_indices = [i + 1 for i, d in enumerate(src.descriptions)
                                if d in desired_bands]

                missing_bands = set(desired_bands) - {
                    src.descriptions[i - 1] for i in band_indices}
                if missing_bands:
                    raise ValueError(f"Missing bands {missing_bands}")

                data = src.read(band_indices)

                # ---- print distribution for the 'missing' mask band ----------
                if desired_bands and desired_bands[-1].lower() == "missing":
                    mask = data[-1]                         # last band read
                    vals, counts = np.unique(mask, return_counts=True)
                    dist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
                    print(f"    Missing‑band distribution "
                          f"[{_acq_time(image_path.stem)}]: {dist}")

                # ---- write filtered stack ------------------------------------
                meta = src.meta.copy()
                meta.update({"count": len(band_indices), "dtype": data.dtype})

                out_path = output_dir / f"{image_path.stem}_filtered.tif"
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(data)
                    dst.descriptions = tuple(desired_bands)

                print(f"    ✔ Saved → {out_path.name}\n")

        except (RasterioIOError, ValueError) as err:
            print(f"⚠ {image_path.name}: {err}")
        except Exception as err:
            print(f"⚠ Unexpected error with {image_path.name}: {err}")

def create_vrt(file_list, vrt_path, combined_dir=None):
    """
    Creates a VRT (Virtual Raster) from a list of raster files,
    then optionally reads it into a NumPy array and saves as .npy.

    Parameters:
      - file_list: List of file paths to .tif files.
      - vrt_path: Path where the VRT file will be saved.
      - combined_dir: Optional Path to directory where .npy should be saved.
    Returns:
      - vrt_path: Path to the created VRT.
    """
    file_list = [str(fp) for fp in file_list]
    vrt_path = Path(vrt_path)

    # 1) build temporary file list
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for file_path in file_list:
            temp_file.write(f"{file_path}\n")
        temp_name = temp_file.name

    # 2) run gdalbuildvrt
    cmd = [
        'gdalbuildvrt',
        '-input_file_list', temp_name,
        str(vrt_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"✔ VRT created at {vrt_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating VRT: {e.stderr.decode()}")
        raise
    finally:
        os.remove(temp_name)

    # 3) optional: read VRT and save .npy
    if combined_dir is not None:
        combined_dir = Path(combined_dir)
        combined_dir.mkdir(parents=True, exist_ok=True)
        npy_path = combined_dir / f"{vrt_path.stem}.npy"

        with rasterio.open(vrt_path) as src:
            arr = src.read()

        np.save(npy_path, arr)
        print(f"✔ NumPy array saved at {npy_path}")

    return vrt_path


# 
