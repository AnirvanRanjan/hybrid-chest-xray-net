"""
01_build_cache.py  —  run ONCE, on a Kaggle CPU session (no GPU needed).

Reads the full-resolution NIH ChestX-ray14 images (1024x1024) and writes a 384x384
grayscale copy of every image. Training at 224 and fine-tuning at 320 then *downsample*
from 384, so the 320 stage really contains more detail than the 224 stage (the old
pipeline upsampled 224px images to 320).

The cache is ONE uncompressed zip (nih384.zip, ~10 GB) instead of 112,120 loose files:
Kaggle saves notebook output file by file, and a huge file count makes the commit crawl
or fail. 02_train.py reads images straight from the zip.

Kaggle setup:
  1. New notebook -> Add Input -> dataset "nih-chest-xrays/data" (NIH Chest X-rays, ~45 GB)
  2. Accelerator: None (CPU). Internet: off is fine.
  3. !python /kaggle/input/<path to retrain-code>/01_build_cache.py
     (the NIH folder is found automatically; pass --src to override)
  4. Save Version (Save & Run All). The output (nih384.zip + 3 metadata files) is the
     only input the training notebook needs.
"""
import argparse
import glob
import os
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2

SIZE = 384
META = ["Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"]


def find_src(root="/kaggle/input"):
    hits = [os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "Data_Entry_2017.csv"), recursive=True)
            if glob.glob(os.path.join(os.path.dirname(p), "images_*"))]
    if len(hits) != 1:
        raise SystemExit(f"Could not find exactly one NIH folder under {root} (found {hits}); pass --src")
    return hits[0]


def encode(path):
    """Read, downsample with area interpolation, re-encode as PNG. Returns (name, bytes or None)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # a few NIH files are RGBA; this flattens them
    if img is None:
        return os.path.basename(path), None
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return os.path.basename(path), buf.tobytes() if ok else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="NIH folder with images_*/ and Data_Entry_2017.csv (default: search /kaggle/input)")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--workers", type=int, default=max(16, 4 * (os.cpu_count() or 1)),
                    help="parallel readers; reading Kaggle's mounted input is the bottleneck, not CPU, "
                         "so more readers than CPUs helps")
    a = ap.parse_args()
    src = a.src or find_src()
    os.makedirs(a.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(src, "images_*", "**", "*.png"), recursive=True))
    print(f"Source: {src}\nFound {len(paths):,} images (expected 112,120); "
          f"{a.workers} workers on {os.cpu_count()} CPUs", flush=True)
    t0, failures, written = time.time(), [], 0
    with zipfile.ZipFile(os.path.join(a.out, "nih384.zip"), "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf, \
            ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (name, data) in enumerate(ex.map(encode, paths, chunksize=64), 1):
            if data is None:
                failures.append(name)
            else:
                zf.writestr(name, data)
                written += 1
            if i % 2000 == 0 or i == len(paths):
                rate = i / (time.time() - t0)
                print(f"{i:,}/{len(paths):,} done, {rate:.0f} img/s, ~{(len(paths) - i) / rate / 60:.0f} min left", flush=True)
    print(f"Done. Images in nih384.zip: {written:,}. Failed reads: {len(failures)} {failures[:10]}")

    # Copy metadata next to the zip so the training notebook needs one input.
    for name in META:
        shutil.copy(os.path.join(src, name), os.path.join(a.out, name))
    print("Copied " + ", ".join(META))
