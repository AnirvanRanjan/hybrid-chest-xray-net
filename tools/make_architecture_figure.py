"""Draws ms_hybrid_architecture.png (Figure 1) from the model as defined in retrain/02_train.py.
Shapes are for a 224x224 input; the 320x320 values are given in brackets.

  python make_architecture_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, EDGE = "#1f2328", "#57606a", "#57606a"
FILL = {"input": "#f6f8fa", "cnn": "#ddeefc", "fpn": "#e3f3e6", "swin": "#fbefd9",
        "head": "#f1e6fb", "cam": "#ffffff"}

fig, ax = plt.subplots(figsize=(15, 4.6))
ax.set_xlim(0, 150)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, title, lines, kind, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=FILL[kind], ec=EDGE, lw=1.1, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 7.4, "\n".join(lines), ha="center", va="top", fontsize=8.6,
            color=INK, linespacing=1.45)


def arrow(x0, y0, x1, y1, label=None, dashed=False, lx=0, ly=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=MUTED, ls="--" if dashed else "-"))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha="center", va="bottom",
                fontsize=7.8, color=MUTED)


Y, H = 14, 26
# 1 input
box(1, Y, 17, H, "Input", ["frontal CXR", "224 × 224", "[320 × 320]", "CLAHE,", "ImageNet norm."], "input")
# 2 DenseNet with three taps
box(23, Y, 22, H, "DenseNet-121", [], "cnn")
ax.text(34, Y + H - 7.4, "ImageNet-1k", ha="center", va="center", fontsize=8.6, color=INK)
arrow(18, Y + H / 2, 23, Y + H / 2)
taps = [("P3  256 × 14 × 14", "after transition2 [20 × 20]", 28.5), ("P4  512 × 7 × 7", "after transition3 [10 × 10]", 23.0),
        ("P5  1024 × 7 × 7", "after norm5 [10 × 10]", 17.5)]
for name, where, ty in taps:
    ax.text(34, ty + 1.3, name, ha="center", va="center", fontsize=8.4, color=INK, family="DejaVu Sans Mono")
    ax.text(34, ty - 1.0, where, ha="center", va="center", fontsize=7.4, color=MUTED)
    arrow(45, ty, 51, ty)
# 3 FPN
box(51, Y, 22, H, "FPN", ["1×1 lateral convs", "to 256 channels", "top-down: nearest", "resize + sum", "",
                          "m3  256 × 14 × 14", "[20 × 20]"], "fpn")
arrow(73, Y + H / 2, 79, Y + H / 2)
# 4 Swin
box(79, Y, 25, H, "Swin-B", ["patch embed. replaced", "by 1×1 conv 256→128", "(one token per m3 cell)",
                             "4 stages: 14, 7, 4, 2", "[20, 10, 5, 3] token grids", "ImageNet-1k",
                             "→ pooled 1024-d"], "swin")
arrow(104, Y + H / 2, 110, Y + H / 2)
# 5 head
box(110, Y, 19, H, "Head", ["LayerNorm(1024)", "Linear 1024→512", "GELU", "Dropout 0.5", "Linear 512→14"], "head")
arrow(129, Y + H / 2, 134, Y + H / 2)
box(134, Y + 5, 15, H - 10, "Output", ["14 sigmoid", "probabilities", "per-class", "thresholds"], "input")
# Grad-CAM on m3
box(49, 1.5, 26, 8.5, "", [], "cam", dashed=True)
ax.text(62, 7.7, "Grad-CAM on m3", ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
ax.text(62, 4.2, "class-probability gradients", ha="center", va="center", fontsize=7.6, color=MUTED)
arrow(62, Y, 62, 10, dashed=True)
ax.text(75, 44.5, "Shapes are channels × height × width for a 224 × 224 input; values in brackets are for the "
        "320 × 320 fine-tuning stage.", ha="center", va="center", fontsize=8.2, color=MUTED)

fig.savefig("ms_hybrid_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote ms_hybrid_architecture.png")
