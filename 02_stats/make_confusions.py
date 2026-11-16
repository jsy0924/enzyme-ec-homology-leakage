"""Per-model confusion matrices, drawn from reproduced out-of-fold predictions.

Outputs
  fig2_confusion_A.png          composition: random vs cluster@0.05
  fig2_confusion_B.png          ESM-2 8M:    random vs cluster@0.05
  fig2_confusion_C.png          ESM-2 650M:  random vs cluster@0.05
  fig2_confusion_strict_ABC.png A / B / C side by side at cluster@0.05
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

WORK = Path("/home/claude/analysis")
OUT = WORK / "figs"
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
CMAP = "Blues"                       # sequential: one hue, light -> dark
NAMES = {"A": "A: amino-acid composition",
         "B": "B: ESM-2 8M",
         "C": "C: ESM-2 650M"}
STRICT = "cluster@0.05"
TAGS = {"random": "random split (leakage)",
        STRICT: "cluster split, J < 0.05"}

P = np.load(WORK / "preds.npz")
y = P["y"]
LABELS = sorted(np.unique(y).tolist())


def draw(ax, model, split, show_ylabel=True):
    p = P[f"pred__{model}__{split}"]
    cm = confusion_matrix(y, p, labels=LABELS, normalize="true")
    im = ax.imshow(cm, cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels([f"EC{l}" for l in LABELS], fontsize=9.5, color=INK)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels([f"EC{l}" for l in LABELS], fontsize=9.5, color=INK)
    ax.set_xlabel("predicted", fontsize=10, color=INK_2)
    if show_ylabel:
        ax.set_ylabel("true", fontsize=10, color=INK_2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-.5, len(LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(LABELS), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = cm[i, j]
            if i == j or v >= 0.05:          # diagonal + notable confusions only
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.6, fontweight="bold" if i == j else "normal",
                        color="#ffffff" if v > 0.55 else INK)
    f1 = f1_score(y, p, average="macro")
    return im, f1


def pair_figure(model):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.3), facecolor=SURFACE)
    for k, (ax, split) in enumerate(zip(axes, ["random", STRICT])):
        ax.set_facecolor(SURFACE)
        im, f1 = draw(ax, model, split, show_ylabel=(k == 0))
        ax.set_title(f"{TAGS[split]}\nmacro-F1 = {f1:.3f}",
                     fontsize=11, color=INK, pad=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cb.ax.tick_params(labelsize=9, length=0, colors=INK_2)
    cb.outline.set_visible(False)
    cb.set_label("fraction of true class (row-normalised)", fontsize=9.5,
                 color=INK_2)
    fig.suptitle(NAMES[model], fontsize=14, fontweight="bold", color=INK,
                 x=0.06, ha="left", y=0.99)
    path = OUT / f"fig2_confusion_{model}.png"
    fig.savefig(path, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def strict_figure():
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.4), facecolor=SURFACE)
    for k, (ax, model) in enumerate(zip(axes, "ABC")):
        ax.set_facecolor(SURFACE)
        im, f1 = draw(ax, model, STRICT, show_ylabel=(k == 0))
        ax.set_title(f"{NAMES[model]}\nmacro-F1 = {f1:.3f}",
                     fontsize=11, color=INK, pad=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.017, pad=0.02)
    cb.ax.tick_params(labelsize=9, length=0, colors=INK_2)
    cb.outline.set_visible(False)
    cb.set_label("fraction of true class (row-normalised)", fontsize=9.5,
                 color=INK_2)
    fig.suptitle("Same strict split (cluster, 4-mer Jaccard < 0.05), three representations",
                 fontsize=14, fontweight="bold", color=INK, x=0.05, ha="left",
                 y=1.01)
    path = OUT / "fig2_confusion_strict_ABC.png"
    fig.savefig(path, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for m in "ABC":
        print("saved", pair_figure(m))
    print("saved", strict_figure())
