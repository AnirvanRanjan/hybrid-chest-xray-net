"""
02_train.py  —  corrected training + evaluation for the MS-Hybrid CXR paper.

What changed versus the original notebook / patent appendix
  * A patient-wise VALIDATION split is carved out of train_val_list.txt.
    Early stopping, LR scheduling, checkpoint choice and thresholds use it.
    The official test list is evaluated exactly once, at the very end.
  * Sampler bug fixed. The old weight (labels * w).sum() gave "No Finding"
    images weight 0, so they were never sampled. Options now:
      --sampler none   plain shuffling (ASL alone handles imbalance)
      --sampler floor  per-image weight = max_k over positive classes of
                       1/(sqrt(n_k)+1); images with no positive label get the
                       smallest class weight (this is the paper's Eq. 8).
  * Images come from the 384px cache of the full-resolution dataset, so the
    320px stage is a real resolution increase.
  * Labels are parsed by splitting "Finding Labels" on "|" (exact match).
  * Checkpoint + resume, and a time guard for Kaggle's 12-hour sessions.
  * Per-image probabilities for val and test are saved with Image Index,
    Patient ID, age and sex, for the bootstrap / calibration / subgroup work.

Run examples (Kaggle, GPU T4 x1 or P100):
  # Stage 1, 224px hybrid, floor sampler
  !python 02_train.py --model hybrid --sampler floor --stage 224 --run_name hyb_floor
  # Stage 2, 320px fine-tune from the stage-1 best checkpoint
  !python 02_train.py --model hybrid --sampler floor --stage 320 \
        --init /kaggle/working/hyb_floor/best.pt --run_name hyb_floor_320
  # Resume a run that hit the time limit (attach the previous output as input)
  !python 02_train.py ... --resume /kaggle/input/<prev-output>/hyb_floor/last.pt
"""
import argparse
import glob
import io
import json
import os
import random
import shutil
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, Swin_B_Weights, densenet121, swin_b

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(CLASSES)
IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="auto",
                   help="folder holding Data_Entry_2017.csv, the two list files and nih384.zip (or nih384/); "
                        "'auto' searches /kaggle/input")
    p.add_argument("--out_dir", default="/kaggle/working")
    p.add_argument("--run_name", required=True)
    p.add_argument("--model", choices=["hybrid", "densenet"], default="hybrid")
    p.add_argument("--loss", choices=["asl", "bce"], default="asl")
    p.add_argument("--sampler", choices=["none", "floor"], default="floor")
    p.add_argument("--stage", type=int, choices=[224, 320], default=224)
    p.add_argument("--init", default=None, help="checkpoint to start the 320 stage from")
    p.add_argument("--resume", default=None, help="last.pt of an interrupted run")
    p.add_argument("--val_frac", type=float, default=0.10, help="fraction of train_val PATIENTS used for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tta", action="store_true", help="3-view TTA for validation/test (stage 320)")
    p.add_argument("--recipe", choices=["paper", "tuned"], default="paper",
                   help="hybrid training schedule. paper: runs A, B, E (backbone/Swin LR 1e-6, 7 frozen epochs, "
                        "320 LR 2e-6). tuned: fixed 22 Oct 2026 before running -- 1 warm-up epoch training only the "
                        "new layers (FPN, patch embedding, head), then everything; DenseNet LR 1e-4, Swin LR 5e-5, "
                        "new layers 1e-4; gradient clipping 1.0; 320 LR 1e-5")
    p.add_argument("--max_hours", type=float, default=11.0,
                   help="stop cleanly after the epoch that crosses this, to beat Kaggle's 12h limit")
    p.add_argument("--debug_n", type=int, default=0, help="use only N images per split (smoke test)")
    a = p.parse_args()
    if a.data_dir == "auto":
        a.data_dir = find_data_dir()
    if a.epochs is None:
        a.epochs = 30 if a.stage == 224 else 5
    if a.batch_size is None:
        a.batch_size = 32 if a.stage == 224 else 8
    return a


def find_data_dir(root="/kaggle/input"):
    """The cache notebook's output: Data_Entry_2017.csv next to nih384.zip or nih384/."""
    hits = [os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "Data_Entry_2017.csv"), recursive=True)
            if glob.glob(os.path.join(os.path.dirname(p), "nih384*"))]
    if len(hits) != 1:
        raise SystemExit(f"Could not find exactly one cache folder under {root} (found {hits}); pass --data_dir")
    print(f"Using data_dir {hits[0]}")
    return hits[0]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Data: labels, split, sampler weights (pure pandas / numpy)
# ----------------------------------------------------------------------------
def load_metadata(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "Data_Entry_2017.csv"))
    findings = df["Finding Labels"].astype(str).str.split("|")
    for c in CLASSES:
        df[c] = findings.apply(lambda labs, c=c: int(c in labs))
    # A handful of NIH ages are impossible (e.g. 148, 412). Mark them missing.
    age = pd.to_numeric(df["Patient Age"], errors="coerce")
    df["age_clean"] = age.where(age <= 100)
    df["sex"] = df["Patient Gender"]
    return df


def read_list(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def make_splits(df, data_dir, val_frac, seed):
    trainval = read_list(os.path.join(data_dir, "train_val_list.txt"))
    test = read_list(os.path.join(data_dir, "test_list.txt"))
    assert not trainval & test, "train_val_list and test_list overlap"
    assert trainval | test == set(df["Image Index"]), "list files do not cover Data_Entry_2017.csv exactly"
    tv = df[df["Image Index"].isin(trainval)]
    te = df[df["Image Index"].isin(test)]

    patients = np.array(sorted(tv["Patient ID"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = int(round(len(patients) * val_frac))
    val_pat = set(patients[:n_val])

    va = tv[tv["Patient ID"].isin(val_pat)]
    tr = tv[~tv["Patient ID"].isin(val_pat)]

    # Sanity checks: no patient in two partitions.
    assert not set(tr["Patient ID"]) & set(va["Patient ID"])
    assert not set(tr["Patient ID"]) & set(te["Patient ID"])
    assert not set(va["Patient ID"]) & set(te["Patient ID"])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def sampler_weights(labels):
    """Per-image weight = max over the image's positive classes of 1/(sqrt(n_k)+1).
    Images with no positive label (No Finding) get min_k 1/(sqrt(n_k)+1)."""
    pos = labels.sum(axis=0)
    w_cls = 1.0 / (np.sqrt(pos) + 1.0)
    masked = np.where(labels == 1, w_cls[None, :], -np.inf)
    w_img = masked.max(axis=1)
    w_img[~np.isfinite(w_img)] = w_cls.min()
    return w_img


def search_thresholds(y_true, y_prob, grid=np.round(np.arange(0.0, 1.0001, 0.01), 2)):
    """Per-class F1-maximising threshold on VALIDATION predictions; rule is prob >= t."""
    th = np.zeros(y_true.shape[1])
    for k in range(y_true.shape[1]):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y_true[:, k], (y_prob[:, k] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        th[k] = best_t
    return th


def macro_auroc(y_true, y_prob):
    aucs = [roc_auc_score(y_true[:, k], y_prob[:, k])
            for k in range(y_true.shape[1]) if len(np.unique(y_true[:, k])) > 1]
    return float(np.mean(aucs))


# ----------------------------------------------------------------------------
# Transforms and dataset
# ----------------------------------------------------------------------------
class CLAHETransform:
    def __init__(self, clip_limit=1.5, grid_size=(8, 8)):
        self.clip_limit, self.grid_size, self.clahe = clip_limit, grid_size, None

    def __getstate__(self):   # cv2 objects can't be pickled (spawned DataLoader workers)
        return {**self.__dict__, "clahe": None}

    def __call__(self, img):
        if self.clahe is None:
            self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        return Image.fromarray(cv2.cvtColor(self.clahe.apply(gray), cv2.COLOR_GRAY2RGB))


def build_transforms(size, train):
    t = [transforms.Resize((size, size)), CLAHETransform()]
    if train:
        t += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10 if size == 224 else 15)]
        if size == 320:
            t += [transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    t += [transforms.ToTensor(), transforms.Normalize(IMNET_MEAN, IMNET_STD)]
    return transforms.Compose(t)


class ImageSource:
    """Cached 384px images, from data_dir/nih384.zip (one file; what 01_build_cache.py writes)
    or from a folder of PNGs (data_dir/nih384/, or a zip that Kaggle extracted into a folder)."""
    def __init__(self, data_dir):
        z = os.path.join(data_dir, "nih384.zip")
        self.zip_path = z if os.path.isfile(z) else None
        self.dir = z if os.path.isdir(z) else os.path.join(data_dir, "nih384")
        self._zf, self._pid = None, None

    def __getstate__(self):   # DataLoader workers open their own zip handle
        return {**self.__dict__, "_zf": None, "_pid": None}

    def open(self, name):
        """Missing image -> crash loudly (no silent zero images)."""
        if self.zip_path is None:
            return Image.open(os.path.join(self.dir, name)).convert("L")
        if self._pid != os.getpid():
            self._zf, self._pid = zipfile.ZipFile(self.zip_path), os.getpid()
        return Image.open(io.BytesIO(self._zf.read(name))).convert("L")


class CXRDataset(Dataset):
    def __init__(self, df, source, transform):
        self.names = df["Image Index"].tolist()
        self.labels = df[CLASSES].values.astype(np.float32)
        self.source, self.transform = source, transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = self.source.open(self.names[i])
        return self.transform(img), torch.from_numpy(self.labels[i])


# ----------------------------------------------------------------------------
# Models (architecture identical to the paper / patent appendix)
# ----------------------------------------------------------------------------
class DenseNetBackbone(nn.Module):
    """Taps after transition2, transition3 and norm5:
       P3 = 256 x H/16, P4 = 512 x H/32, P5 = 1024 x H/32  (224 input: 14x14, 7x7, 7x7)."""
    def __init__(self):
        super().__init__()
        f = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).features
        self.stem = nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0)
        self.block1 = nn.Sequential(f.denseblock1, f.transition1)
        self.block2 = nn.Sequential(f.denseblock2, f.transition2)
        self.block3 = nn.Sequential(f.denseblock3, f.transition3)
        self.block4 = nn.Sequential(f.denseblock4, f.norm5)

    def forward(self, x):
        p3 = self.block2(self.block1(self.stem(x)))
        p4 = self.block3(p3)
        p5 = self.block4(p4)
        return p3, p4, p5


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lat5 = nn.Conv2d(1024, 256, 1)
        self.lat4 = nn.Conv2d(512, 256, 1)
        self.lat3 = nn.Conv2d(256, 256, 1)

    def forward(self, p3, p4, p5):
        m5 = self.lat5(p5)
        m4 = self.lat4(p4) + F.interpolate(m5, size=p4.shape[-2:], mode="nearest")
        m3 = self.lat3(p3) + F.interpolate(m4, size=p3.shape[-2:], mode="nearest")
        return m3


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.fpn = DenseNetBackbone(), FPN()
        swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        swin.features[0][0] = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        swin.head = nn.Identity()
        self.transformer = swin
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 512), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        return self.classifier(self.transformer(self.fpn(p3, p4, p5)))


class DenseNetBaseline(nn.Module):
    """CheXNet-style DenseNet-121 with a 14-way linear head."""
    def __init__(self):
        super().__init__()
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(1024, NUM_CLASSES)
        self.net = m

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    """ASL (Ridnik et al., 2021), written to match the paper's Eq. 7: the negative term is
    weighted by p_m^gamma_neg with p_m = max(p - clip, 0). (The old notebook weighted it by the
    unshifted p^gamma_neg.) gamma_pos=0, gamma_neg=4, clip=0.05 for stage 224.
    Stage 320 uses gamma_neg=3, gamma_pos=1, label smoothing 0.05, logit clamp [-10, 10]."""
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, smoothing=0.0, logit_clamp=None, eps=1e-6):
        super().__init__()
        self.gn, self.gp, self.clip, self.s, self.lc, self.eps = gamma_neg, gamma_pos, clip, smoothing, logit_clamp, eps

    def forward(self, x, y):
        x = x.float()
        if self.lc is not None:
            x = x.clamp(-self.lc, self.lc)
        if self.s > 0:
            y = y * (1 - self.s) + 0.5 * self.s
        p = torch.sigmoid(x)
        p_neg = (1 - p + self.clip).clamp(max=1) if self.clip else 1 - p
        loss_pos = y * torch.log(p.clamp(min=self.eps)) * (1 - p).pow(self.gp)
        loss_neg = (1 - y) * torch.log(p_neg.clamp(min=self.eps)) * (1 - p_neg).pow(self.gn)
        return -(loss_pos + loss_neg).mean()


# ----------------------------------------------------------------------------
# Train / eval helpers
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    out = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(imgs).float())
            if tta:
                p_flip = torch.sigmoid(model(torch.flip(imgs, dims=[3])).float())
                big = F.interpolate(imgs, scale_factor=1.1, mode="bilinear", align_corners=False)
                p_big = torch.sigmoid(model(big).float())
                p = (p + p_flip + p_big) / 3.0
        out.append(p.cpu().numpy())
    return np.vstack(out)


def set_phase(model, epoch, args):
    """Hybrid unfreezing (stage 224 only). Epochs are 0-indexed here.
       paper: epochs 1-5 head+FPN (the new patch embedding stays frozen at its random init, as in runs A/B),
              epochs 6-7 + Swin, epoch 8+ everything.
       tuned: epoch 1 new layers only (FPN, patch embedding, head), epoch 2+ everything."""
    if args.model != "hybrid" or args.stage != 224:
        for p in model.parameters():
            p.requires_grad = True
        return
    if args.recipe == "tuned":
        train_swin = train_cnn = epoch >= 1
    else:
        train_swin, train_cnn = epoch >= 5, epoch >= 7
    for p in model.backbone.parameters():
        p.requires_grad = train_cnn
    for p in model.transformer.parameters():
        p.requires_grad = train_swin
    if args.recipe == "tuned":
        for p in model.transformer.features[0][0].parameters():   # new 1x1 patch embedding: always trained
            p.requires_grad = True


def build_optimizer(model, args):
    if args.stage == 320:
        return torch.optim.AdamW(model.parameters(), lr=1e-5 if args.recipe == "tuned" else 2e-6, weight_decay=1e-4)
    if args.model == "hybrid" and args.recipe == "tuned":
        embed = list(model.transformer.features[0][0].parameters())
        embed_ids = {id(p) for p in embed}
        return torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-4},
            {"params": [p for p in model.transformer.parameters() if id(p) not in embed_ids], "lr": 5e-5},
            {"params": embed + list(model.fpn.parameters()) + list(model.classifier.parameters()), "lr": 1e-4},
        ], weight_decay=1e-4)
    if args.model == "hybrid":
        return torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-6},
            {"params": model.transformer.parameters(), "lr": 1e-6},
            {"params": model.fpn.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ], weight_decay=1e-4)
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


def save_predictions(df, probs, path):
    out = df[["Image Index", "Patient ID", "age_clean", "sex", "View Position"] + CLASSES].copy()
    out = out.rename(columns={c: f"y_{c}" for c in CLASSES})
    for k, c in enumerate(CLASSES):
        out[f"p_{c}"] = probs[:, k]
    out.to_csv(path, index=False)


def per_class_report(y, p, th):
    rows = []
    for k, c in enumerate(CLASSES):
        b = (p[:, k] >= th[k]).astype(int)
        rows.append({
            "class": c,
            # NaN when a class has only one label value (happens in --debug_n subsets)
            "auroc": roc_auc_score(y[:, k], p[:, k]) if len(np.unique(y[:, k])) > 1 else np.nan,
            "f1": f1_score(y[:, k], b, zero_division=0),
            "precision": precision_score(y[:, k], b, zero_division=0),
            "recall": recall_score(y[:, k], b, zero_division=0),
            "threshold": th[k],
            "n_pos": int(y[:, k].sum()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = get_args()
    t_start = time.time()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    df = load_metadata(args.data_dir)
    tr, va, te = make_splits(df, args.data_dir, args.val_frac, args.seed)
    if args.debug_n:
        tr, va, te = tr.head(args.debug_n), va.head(args.debug_n), te.head(args.debug_n)
    split_info = {
        "train_images": len(tr), "train_patients": int(tr["Patient ID"].nunique()),
        "val_images": len(va), "val_patients": int(va["Patient ID"].nunique()),
        "test_images": len(te), "test_patients": int(te["Patient ID"].nunique()),
    }
    print(split_info)
    with open(os.path.join(run_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])[
        ["Image Index", "Patient ID", "split"]].to_csv(os.path.join(run_dir, "split.csv"), index=False)
    # Positive counts per class and split (plus No Finding) for the paper's dataset table.
    counts = {name: {**{c: int(d[c].sum()) for c in CLASSES},
                     "No Finding": int((d[CLASSES].sum(axis=1) == 0).sum()), "images": len(d)}
              for name, d in [("all", df), ("train", tr), ("val", va), ("test", te)]}
    pd.DataFrame(counts).rename_axis("class").to_csv(os.path.join(run_dir, "split_class_counts.csv"))

    source = ImageSource(args.data_dir)
    size = args.stage
    tr_ds = CXRDataset(tr, source, build_transforms(size, train=True))
    va_ds = CXRDataset(va, source, build_transforms(size, train=False))
    te_ds = CXRDataset(te, source, build_transforms(size, train=False))

    g = torch.Generator().manual_seed(args.seed)
    if args.sampler == "floor":
        w = torch.as_tensor(sampler_weights(tr_ds.labels), dtype=torch.double)
        tr_dl = DataLoader(tr_ds, batch_size=args.batch_size,
                           sampler=WeightedRandomSampler(w, len(w), replacement=True, generator=g),
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = (HybridModel() if args.model == "hybrid" else DenseNetBaseline()).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
        print(f"Initialised from {args.init}")

    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.stage == 224:
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    else:
        criterion = AsymmetricLoss(gamma_neg=3, gamma_pos=1, clip=0.05, smoothing=0.05, logit_clamp=10)

    optimizer = build_optimizer(model, args)
    if args.stage == 224:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best_auc, patience, history = 0, -1.0, 0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_auc, patience, history = ck["epoch"] + 1, ck["best_auc"], ck["patience"], ck["history"]
        prev_best = os.path.join(os.path.dirname(args.resume), "best.pt")
        if os.path.exists(prev_best) and os.path.abspath(prev_best) != os.path.abspath(os.path.join(run_dir, "best.pt")):
            shutil.copy(prev_best, os.path.join(run_dir, "best.pt"))
        g.manual_seed(args.seed + start_epoch)   # don't replay epoch 1's sample order
        print(f"Resumed at epoch {start_epoch + 1}, best val AUROC {best_auc:.4f}")

    use_tta = args.tta and args.stage == 320
    epoch_hours = 0.0
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        set_phase(model, epoch, args)
        model.train()
        run_loss, n = 0.0, 0
        for imgs, lbls in tr_dl:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(imgs)
            loss = criterion(logits.float(), lbls)
            scaler.scale(loss).backward()
            if args.stage == 320 or args.recipe == "tuned":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item(); n += 1

        p_val = predict(model, va_dl, device, tta=use_tta)
        val_auc = macro_auroc(va_ds.labels, p_val)
        val_f1 = f1_score(va_ds.labels, (p_val >= 0.5).astype(int), average="macro", zero_division=0)
        if args.stage == 224:
            scheduler.step(val_auc)
        else:
            scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": run_loss / max(n, 1),
                        "val_auroc": val_auc, "val_f1_at_0.5": val_f1,
                        "lr": optimizer.param_groups[-1]["lr"]})
        print(history[-1], flush=True)

        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auroc": val_auc},
                       os.path.join(run_dir, "best.pt"))
        else:
            patience += 1

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_auc": best_auc, "patience": patience,
                    "history": history},
                   os.path.join(run_dir, "last.pt"))
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

        if args.stage == 224 and patience >= args.patience:
            print("Early stopping on validation AUROC.")
            break
        epoch_hours = max(epoch_hours, (time.time() - t_epoch) / 3600)
        if epoch + 1 < args.epochs and (time.time() - t_start) / 3600 + epoch_hours > args.max_hours:
            print(f"Time guard hit after epoch {epoch + 1}. Save Version, then rerun with --resume.")
            return

    # ---------------- Final evaluation: test set touched once ----------------
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device)["model"])
    p_val = predict(model, va_dl, device, tta=use_tta)
    p_test = predict(model, te_dl, device, tta=use_tta)
    th = search_thresholds(va_ds.labels, p_val)

    save_predictions(va, p_val, os.path.join(run_dir, "pred_val.csv"))
    save_predictions(te, p_test, os.path.join(run_dir, "pred_test.csv"))
    np.save(os.path.join(run_dir, "thresholds_val.npy"), th)

    rep = per_class_report(te_ds.labels, p_test, th)
    rep.to_csv(os.path.join(run_dir, "test_per_class.csv"), index=False)
    summary = {
        "run": args.run_name, "best_val_auroc": best_auc,
        "test_macro_auroc": float(np.nanmean(rep["auroc"])),
        "test_macro_f1_at_val_thresholds": float(rep["f1"].mean()),
        "test_macro_f1_at_0.5": float(f1_score(te_ds.labels, (p_test >= 0.5).astype(int),
                                               average="macro", zero_division=0)),
        "tta": use_tta, **split_info,
    }
    with open(os.path.join(run_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(rep.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))

    # Everything except checkpoints in one file, ready to download from Kaggle's Output tab.
    # (Written only when the run completes; the .pt files stay in the output for --init/--resume.)
    zip_path = os.path.join(args.out_dir, f"{args.run_name}_results.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(run_dir)):
            if not name.endswith(".pt"):
                zf.write(os.path.join(run_dir, name), arcname=os.path.join(args.run_name, name))
    print(f"Results (no checkpoints) zipped to {zip_path}")


if __name__ == "__main__":
    main()
