"""
04_gradcam.py  —  Grad-CAM figure for the paper, hooked on the FPN output m3 (model.fpn),
the layer the manuscript describes. (The old notebook hooked DenseNet norm5.)

Examples are chosen by a rule fixed in advance, not by eye: for each requested class, draw
--n_per_class images at random (seeded) from the TEST true positives at the validation
threshold. The chosen images are listed in gradcam_examples.csv; say this in the caption.

Runs on CPU. Needs the run folder (args.json, pred_test.csv, thresholds_val.npy), the
checkpoint, and the nih384 cache (nih384.zip or nih384/) (a Kaggle CPU session with the cache
attached works; so does copying the few listed images locally).

  python 04_gradcam.py --run_dir results/hyb_320 --ckpt results/hyb_320/best.pt \
      --data_dir /kaggle/input/nih384-cache --classes Cardiomegaly Effusion Pneumothorax \
      --out ../paper/generated/E
"""
import argparse
import importlib.util
import json
import os
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location("train", os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_train.py"))
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def grad_cam(model, x, k):
    """Heatmap (H_in x W_in, 0..1) for class k from the FPN output m3, plus the predicted probability."""
    store = {}

    def hook(_, __, out):
        out.retain_grad()
        store["A"] = out

    h = model.fpn.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    prob = torch.sigmoid(model(x).float())[0, k]
    prob.backward()
    h.remove()
    A, G = store["A"], store["A"].grad                      # 1 x 256 x h x w
    alpha = G.mean(dim=(2, 3), keepdim=True)                # Eq. 9: GAP of the gradients
    cam = F.relu((alpha * A).sum(dim=1, keepdim=True))      # Eq. 10
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.detach().numpy(), prob.item(), tuple(A.shape[-2:])


def overlay(gray, cam):
    """0.6 * image + 0.4 * JET(heatmap), as in the paper."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.uint8(np.clip(0.6 * rgb + 0.4 * heat, 0, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="auto", help="cache folder (nih384.zip or nih384/); 'auto' searches /kaggle/input")
    ap.add_argument("--classes", nargs="+", default=["Cardiomegaly", "Effusion", "Pneumothorax"])
    ap.add_argument("--n_per_class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gradcam")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    run_args = json.load(open(os.path.join(a.run_dir, "args.json")))
    assert run_args["model"] == "hybrid", "Grad-CAM on m3 needs the hybrid model"
    size = run_args["stage"]
    model = train.HybridModel()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    model.eval()
    tf = train.build_transforms(size, train=False)
    show = train.transforms.Compose([train.transforms.Resize((size, size)), train.CLAHETransform()])

    pred = pd.read_csv(os.path.join(a.run_dir, "pred_test.csv"))
    th = np.load(os.path.join(a.run_dir, "thresholds_val.npy"))
    source = train.ImageSource(train.find_data_dir() if a.data_dir == "auto" else a.data_dir)
    rng = np.random.default_rng(a.seed)
    rows = []
    for c in a.classes:
        k = train.CLASSES.index(c)
        tp = pred[(pred[f"y_{c}"] == 1) & (pred[f"p_{c}"] >= th[k])]
        if len(tp) == 0:
            print(f"{c}: no true positives at the validation threshold; skipped")
            continue
        pick = tp.iloc[rng.choice(len(tp), size=min(a.n_per_class, len(tp)), replace=False)]
        for _, r in pick.iterrows():
            img = source.open(r["Image Index"])
            cam, prob, grid = grad_cam(model, tf(img).unsqueeze(0), k)
            gray = np.array(show(img).convert("L"))
            others = [o for o in train.CLASSES if r[f"y_{o}"] == 1 and o != c]
            rows.append({"class": c, "Image Index": r["Image Index"], "prob": prob, "threshold": th[k],
                         "other_labels": "|".join(others), "cam_grid": f"{grid[0]}x{grid[1]}",
                         "n_true_positives": len(tp), "gray": gray, "overlay": overlay(gray, cam)})

    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)), squeeze=False)
    for (ax0, ax1), r in zip(axes, rows):
        ax0.imshow(r["gray"], cmap="gray"); ax1.imshow(r["overlay"])
        extra = f" (+ {r['other_labels'].replace('|', ', ').replace('_', ' ')})" if r["other_labels"] else ""
        ax0.set_title(textwrap.fill(f"Label: {r['class'].replace('_', ' ')}{extra}", 42), fontsize=8)
        ax1.set_title(f"p = {r['prob']:.2f} (threshold {r['threshold']:.2f})", fontsize=8)
        for ax in (ax0, ax1):
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "gradcam.png"), dpi=300, facecolor="white")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("gray", "overlay")} for r in rows]).to_csv(
        os.path.join(a.out, "gradcam_examples.csv"), index=False)
    print(f"Grad-CAM on model.fpn (m3 grid {rows[0]['cam_grid'] if rows else '?'} at {size}px); "
          f"{len(rows)} examples -> {a.out}/gradcam.png")


if __name__ == "__main__":
    main()
