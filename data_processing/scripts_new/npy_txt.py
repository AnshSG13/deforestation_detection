#!/usr/bin/env python3
"""
compute_stats_and_create_txt_fixed.py

A drop‑in replacement for the original script with the following corrections:

1. **Initialises `std` safely** – the variable is now derived from the aggregated
   statistics instead of being conditionally created.
2. **Removes the bogus `if split == "train"` check after the loop** – statistics
   are computed for the training split immediately after the global pass using
   the numerically stable one‑pass formula `Var = E[X²] – E[X]²`.
3. **No second data pass is required**, which is faster and avoids the stale
   `before_files` pointer problem.
4. Minor clean‑ups and type annotations for clarity.

Usage remains identical to the original script.
"""

from __future__ import annotations

import csv
import gc
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Set

import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Helper: soft timeout context manager (Unix‑only SIGALRM)
# ────────────────────────────────────────────────────────────────────────────────

class TimeoutException(Exception):
    """Raised when a block exceeds the allotted time limit."""


@contextmanager
def time_limit(seconds: int):
    """Limit the execution time of the wrapped block on Unix systems."""

    def _handler(signum, frame):  # noqa: D401  (simple name)
        raise TimeoutException("Timed out!")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:  # Windows & friends – best‑effort (no timeout)
        yield


# ────────────────────────────────────────────────────────────────────────────────
# Core function
# ────────────────────────────────────────────────────────────────────────────────

def compute_stats_and_create_txt_band2_3_4_8_single_folder(
    input_root: str | Path,
    *,
    skip_files: Iterable[int] | None = None,
    timeout_seconds: int = 75,
) -> None:
    """Process train/test/val folders and output stats + id lists.

    Parameters
    ----------
    input_root : Path‑like
        Folder that contains the three `train|test|val/before` sub‑folders.
    skip_files : Iterable[int] | None, optional
        1‑based indices inside each split to skip completely.
    timeout_seconds : int, optional
        Per‑file soft timeout (Unix‑only).
    """

    root = Path(input_root)
    root.mkdir(parents=True, exist_ok=True)

    skip_set: Set[int] = set(skip_files or [])

    # Global accumulators (training split only)
    sum_ = np.zeros(4, dtype=np.float64)
    sumsq = np.zeros(4, dtype=np.float64)
    min_ = np.full(4, np.inf)
    max_ = np.full(4, -np.inf)
    valid_cnt = np.zeros(4, dtype=np.int64)
    missing_cnt = np.zeros(4, dtype=np.int64)

    # Track whether the train split was actually seen
    saw_train = False

    for split in ("train", "test", "val"):
        start = time.time()
        before_dir = root / split / "before"
        if not before_dir.exists():
            print(f"[WARN] {before_dir} missing – skipping {split} split")
            continue

        files = sorted(before_dir.glob("*.npz"))
        if not files:
            print(f"[WARN] no .npz files in {before_dir} – skipping {split}")
            continue

        print(f"Processing {split} – {len(files)} files…")
        problem_log = root / f"{split}_problematic_files.txt"
        processed_ids: List[str] = []

        for idx, f in enumerate(files, 1):  # 1‑based index matches skip list
            if idx in skip_set:
                print(f"  ⏭  #{idx}/{len(files)} {f.name} (skipped)")
                continue

            try:
                with time_limit(timeout_seconds):
                    arr = np.load(f)["array"] 
            except TimeoutException:
                _log(problem_log, idx, f.name, "timeout", f"{timeout_seconds}s")
                continue
            except Exception as e:  # noqa: BLE001
                _log(problem_log, idx, f.name, "load_error", str(e))
                continue

            if arr.shape[0] != 7:
                _log(problem_log, idx, f.name, "wrong_bands", str(arr.shape))
                continue

            # Quick NaN/Inf sniff on a 10×10 patch per band
            sample = arr[:, :10, :10]
            has_nan_or_inf = np.isnan(sample).any() or np.isinf(sample).any()

            if split == "train":
                saw_train = True
                for b in range(4):
                    band = arr[b]
                    missing = np.isnan(band) | np.isinf(band)
                    missing_cnt[b] += missing.sum()

                    valid = np.where(~missing, band, 0.0)
                    cnt = band.size - missing.sum()
                    if cnt == 0:
                        continue  # all missing

                    sum_[b] += valid.sum(dtype=np.float64)
                    sumsq[b] += np.square(valid, dtype=np.float64).sum(dtype=np.float64)
                    min_[b] = min(min_[b], valid.min())
                    max_[b] = max(max_[b], valid.max())
                    valid_cnt[b] += cnt

            processed_ids.append(f"{idx-1:05d}")

            # Free and occasionally poke GC
            del arr
            if idx % 10 == 0:
                gc.collect()

        # Write the id list for the split
        (root / f"{split}.txt").write_text("\n".join(processed_ids))
        print(
            f"  ↳ completed {split}: {len(processed_ids)}/{len(files)} files in "
            f"{time.time() - start:.1f}s",
        )

    # ── Final statistics for training data ───────────────────────────────────
    if saw_train and np.all(valid_cnt > 0):
        mean = sum_ / valid_cnt
        variance = np.maximum(sumsq / valid_cnt - mean**2, 0.0)  # ensure ≥0
        std = np.sqrt(variance, dtype=np.float64)

        _write_csv(root / "train_aggregated_band_statistics.csv", mean, std, min_, max_, valid_cnt, missing_cnt)

        print("\nAggregated training statistics:")
        _print_stats(mean, std, min_, max_, valid_cnt, missing_cnt)
    else:
        print("[WARN] no train split processed – statistics not written")

    print("\n✅ Done!")


# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

BANDS = ("band2", "band3", "band4", "band8")


def _log(file: Path, idx: int, name: str, err_type: str, msg: str) -> None:
    file.parent.mkdir(exist_ok=True)
    with file.open("a") as fp:
        fp.write(f"{idx},{name},{err_type},{msg}\n")


def _write_csv(
    csv_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    min_: np.ndarray,
    max_: np.ndarray,
    valid_cnt: np.ndarray,
    missing_cnt: np.ndarray,
) -> None:
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=(
                "band",
                "mean",
                "std",
                "min",
                "max",
                "valid_pixel_count",
                "missing_pixel_count",
            ),
        )
        writer.writeheader()
        for i, label in enumerate(BANDS):
            writer.writerow(
                {
                    "band": label,
                    "mean": f"{mean[i]:.6f}",
                    "std": f"{std[i]:.6f}",
                    "min": f"{min_[i]:.6f}",
                    "max": f"{max_[i]:.6f}",
                    "valid_pixel_count": int(valid_cnt[i]),
                    "missing_pixel_count": int(missing_cnt[i]),
                }
            )
    print(f"\n📄 Stats written → {csv_path}")


def _print_stats(
    mean: np.ndarray,
    std: np.ndarray,
    min_: np.ndarray,
    max_: np.ndarray,
    valid_cnt: np.ndarray,
    missing_cnt: np.ndarray,
) -> None:
    for i, label in enumerate(BANDS):
        print(
            f"  {label:5s} | mean={mean[i]:.6f}  std={std[i]:.6f}  "
            f"min={min_[i]:.6f}  max={max_[i]:.6f}  "
            f"valid={valid_cnt[i]}  missing={missing_cnt[i]}",
        )


# ────────────────────────────────────────────────────────────────────────────────
# CLI wrapper
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_stats_and_create_txt_fixed.py /path/to/data [skip_ids]")
        sys.exit(1)

    data_root = sys.argv[1]
    skips: List[int] = []
    if len(sys.argv) > 2:
        try:
            skips = [int(x) for x in sys.argv[2].split(",") if x]
        except ValueError:
            print("⚠️  Invalid skip list – ignoring")

    print(f"Input root: {data_root}\nSkip ids : {skips}\n")
    compute_stats_and_create_txt_band2_3_4_8_single_folder(data_root, skip_files=skips)
