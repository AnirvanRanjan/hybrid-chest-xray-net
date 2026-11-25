"""
09_infer_external.py  —  paper 2: run every frozen NIH model of the grid on the cached CheXpert images.

GPU, inference only. Nothing is fitted here; 09_external.py (CPU) does all the evaluation.

Preprocessing is the NIH test-time pipeline of 02_train.py (its build_transforms(224, train=False)):
384px cache -> Resize 224 -> CLAHE -> gray to RGB -> ImageNet normalisation. Each image is resized and
CLAHE'd once (uint8, 224x224) and normalised on the GPU, which gives the same tensor as the
torchvision pipeline (checked below on the first images).

Sanity check before CheXpert: every checkpoint must reproduce its own saved NIH test predictions
(check_nih.csv, built from results/<run>/pred_test.csv) on the same NIH images; max |diff| must be
below 0.02 (fp16 autocast, other batch size), otherwise the run is skipped and the log says why.

Inputs (anywhere under /kaggle/input, found by name):
  02_train.py              (retrain-code dataset)
  chexpert384.zip          (chexpert-probe output; a zip, or the folder Kaggle extracted it to)
  nih384.zip               (nih384-cache output)
  <run>/best.pt or <run>.pt  for each run in RUNS
  check_nih.csv            (p2-upes-checkpoints dataset)
Outputs in /kaggle/working/ext_preds/: <run>.csv.gz (key = "valid/patientX/studyY/viewZ_frontal",
p_<class> for the 14 NIH classes), nih_check.csv, done.json.

  python 09_infer_external.py --max_hours 1.6
"""
import argparse
import importlib.util
import io
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from PIL import Image

RUNS = [  # (run name, --model); order = cheapest first so a time-guard stop loses the least
    ("dn_bce", "densenet"), ("p2_dn_wbce", "densenet"), ("p2_dn_focal", "densenet"), ("dn_asl", "densenet"),
    ("hyb_tuned", "hybrid"), ("p2_hyb_wbce", "hybrid"), ("p2_hyb_focal", "hybrid"), ("p2_hyb_asl", "hybrid"),
    ("p2_swin_bce", "swin"), ("p2_swin_wbce", "swin"), ("p2_swin_focal", "swin"), ("p2_swin_asl", "swin"),
]
SIZE = 224


def index_inputs(root):
    """One walk over the inputs; the extracted image folders (train/, valid/, *.png) are not descended."""
    found = {"pt": [], "files": {}, "dirs": {}}
    for d, subdirs, files in os.walk(root):
        if os.path.basename(d) in ("chexpert384.zip", "nih384.zip", "nih384"):
            found["dirs"][os.path.basename(d)] = d
            subdirs[:] = []
            continue
        subdirs[:] = [s for s in subdirs if s not in ("train", "valid")]
        for f in files:
            p = os.path.join(d, f)
            if f.endswith(".pt"):
                found["pt"].append(p)
            elif f in ("02_train.py", "chexpert384.zip", "nih384.zip", "check_nih.csv"):
                found["files"].setdefault(f, p)
    return found


def find_ckpt(pts, run):
    hits = [p for p in pts if p.endswith(f"/{run}/best.pt") or os.path.basename(p) == f"{run}.pt"]
    return sorted(hits)[0] if hits else None


class Source:
    """Read PNGs from a zip file or from the folder Kaggle extracted it to."""
    def __init__(self, path):
        self.path, self.zf = path, None

    def names(self):
        if os.path.isfile(self.path):
            return [n for n in zipfile.ZipFile(self.path).namelist() if n.endswith(".png")]
        out = []
        for d, _, fs in os.walk(self.path):
            out += [os.path.relpath(os.path.join(d, f), self.path) for f in fs if f.endswith(".png")]
        return out

    def open(self, name):
        if os.path.isdir(self.path):
            return Image.open(os.path.join(self.path, name)).convert("L")
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.path)
        return Image.open(io.BytesIO(self.zf.read(name))).convert("L")


_W = {}


def _init(src_path, train_py):
    _W["src"] = Source(src_path)
    _W["tr"] = load_train(train_py)
    tr = _W["tr"]
    _W["resize"], _W["clahe"] = tr.transforms.Resize((SIZE, SIZE)), tr.CLAHETransform()


def _prep(name):
    img = _W["clahe"](_W["resize"](_W["src"].open(name)))    # RGB PIL, three equal channels
    return np.asarray(img)[:, :, 0].copy()


def preprocess(src_path, names, train_py, workers):
    out = np.empty((len(names), SIZE, SIZE), np.uint8)
    with Pool(workers, initializer=_init, initargs=(src_path, train_py)) as pool:
        for i, a in enumerate(pool.imap(_prep, names, chunksize=64)):
            out[i] = a
    return out


def load_train(path):
    spec = importlib.util.spec_from_file_location("train02", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def infer(model, u8, device, tr, bs=128):
    """Same as 02_train.predict (autocast, sigmoid of the float logits), from uint8 gray images."""
    mean = torch.tensor(tr.IMNET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(tr.IMNET_STD, device=device).view(1, 3, 1, 1)
    model.eval()
    out = []
    for s in range(0, len(u8), bs):
        x = torch.from_numpy(u8[s:s + bs]).to(device).float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
        x = (x - mean) / std
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
        out.append(p.cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/ext_preds")
    ap.add_argument("--max_hours", type=float, default=1.6)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N CheXpert images")
    ap.add_argument("--runs", nargs="*", default=None, help="subset of run names (debug)")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", flush=True)

    inp = index_inputs(a.input_root)
    train_py = inp["files"]["02_train.py"]
    tr = load_train(train_py)
    assert "SwinBaseline" in open(train_py).read(), "old retrain-code version attached"
    chex = inp["files"].get("chexpert384.zip") or inp["dirs"].get("chexpert384.zip")
    nih = inp["files"].get("nih384.zip") or inp["dirs"].get("nih384.zip") or inp["dirs"].get("nih384")
    check = pd.read_csv(inp["files"]["check_nih.csv"])
    runs = [r for r in RUNS if a.runs is None or r[0] in a.runs]
    ckpts = {r: find_ckpt(inp["pt"], r) for r, _ in runs}
    for r, p in ckpts.items():
        print(f"  {r:14s} {p}")
    missing = [r for r, p in ckpts.items() if p is None]
    assert not missing, f"checkpoints not found: {missing}"

    # NIH check images, then CheXpert
    nih_names = sorted(check["Image Index"].unique())
    nih_u8 = preprocess(nih, nih_names, train_py, a.workers)
    # same tensor as the torchvision test pipeline?
    ref_t = tr.build_transforms(SIZE, train=False)
    src = Source(nih)
    for i in range(3):
        ref = ref_t(src.open(nih_names[i]))
        mine = (torch.from_numpy(nih_u8[i]).float().div(255) - torch.tensor(tr.IMNET_MEAN).view(3, 1, 1)) / torch.tensor(tr.IMNET_STD).view(3, 1, 1)
        assert torch.allclose(ref, mine, atol=1e-6), "preprocessing differs from build_transforms"
    print("preprocessing identical to build_transforms(224, train=False)", flush=True)

    names = sorted(Source(chex).names())
    if a.limit:
        names = names[: a.limit]
    t1 = time.time()
    u8 = preprocess(chex, names, train_py, a.workers)
    keys = [n[:-4] for n in names]
    print(f"CheXpert: {len(names)} images preprocessed in {(time.time() - t1) / 60:.1f} min "
          f"({sum(k.startswith('valid/') for k in keys)} valid)", flush=True)

    done, checks = {}, []
    for run, kind in runs:
        dest = os.path.join(a.out, f"{run}.csv.gz")
        el = (time.time() - t0) / 3600
        if el > a.max_hours:
            print(f"time guard ({el:.2f} h): stopping before {run}", flush=True)
            break
        t2 = time.time()
        model = {"hybrid": tr.HybridModel, "densenet": tr.DenseNetBaseline, "swin": tr.SwinBaseline}[kind]().to(device)
        ck = torch.load(ckpts[run], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        # 1) reproduce the saved NIH test predictions
        c = check[check.run == run].set_index("Image Index").loc[nih_names]
        p_nih = infer(model, nih_u8, device, tr)
        diff = float(np.abs(p_nih - c[[f"p_{k}" for k in tr.CLASSES]].values).max())
        ok = diff < 0.02
        checks.append({"run": run, "ckpt": ckpts[run], "epoch": ck.get("epoch"), "max_abs_diff": diff, "ok": ok})
        pd.DataFrame(checks).to_csv(os.path.join(a.out, "nih_check.csv"), index=False)
        print(f"{run}: NIH check max |diff| = {diff:.5f} {'OK' if ok else 'FAILED, skipped'}", flush=True)
        if not ok:
            continue
        # 2) CheXpert
        p = infer(model, u8, device, tr)
        out = pd.DataFrame(p, columns=[f"p_{k}" for k in tr.CLASSES])
        out.insert(0, "key", keys)
        out.to_csv(dest, index=False, float_format="%.6g")
        done[run] = {"minutes": (time.time() - t2) / 60, "n": len(keys)}
        print(f"{run}: {len(keys)} images in {(time.time() - t2) / 60:.1f} min", flush=True)
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(a.out, "done.json"), "w") as f:
        json.dump({"done": done, "hours": (time.time() - t0) / 3600, "torch": torch.__version__}, f, indent=2)
    print(f"finished {len(done)}/{len(runs)} runs in {(time.time() - t0) / 3600:.2f} h", flush=True)
    sys.exit(0 if len(done) == len(runs) else 1)


if __name__ == "__main__":
    main()
