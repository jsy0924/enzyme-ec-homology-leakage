"""t-SNE small multiples: rows = representation (A, B, C), columns = EC class.
In each panel every sequence is a light gray dot and the sequences of one class
are drawn in a single highlight colour. One hue per panel keeps every class
readable without asking the eye to tell seven colours apart.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = Path("/home/claude/tsne")
d = np.load(W / "tsne_coords.npz")
PUR = json.loads((W / "neighbour_purity.json").read_text())["purity"]
y = d["y"]

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8980"
BG_DOT = "#dcdbd5"
HI = "#2a78d6"
ROWS = [("A", "A: amino-acid composition"), ("B", "B: ESM-2 8M"), ("C", "C: ESM-2 650M")]
EC = {1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase", 4: "Lyase",
      5: "Isomerase", 6: "Ligase", 7: "Translocase"}
F1_STRICT = {"A": 0.372, "B": 0.711, "C": 0.848}
F1_RANDOM = {"A": 0.396, "B": 0.804, "C": 0.936}


def main():
    fig, axes = plt.subplots(3, 7, figsize=(16.5, 7.9), facecolor=SURFACE)
    for r, (m, title) in enumerate(ROWS):
        xy = d[f"xy_{m}"]
        for c, ec in enumerate(range(1, 8)):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            mask = y == ec
            ax.scatter(xy[~mask, 0], xy[~mask, 1], s=1.2, color=BG_DOT, lw=0,
                       rasterized=True)
            ax.scatter(xy[mask, 0], xy[mask, 1], s=2.2, color=HI, lw=0,
                       rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#e6e5df")
            if r == 0:
                ax.set_title(f"EC{ec}\n{EC[ec]}", fontsize=10, color=INK, pad=6)
            if c == 0:
                ax.set_ylabel(f"{title}\n\nmacro-F1 {F1_RANDOM[m]:.3f} (random)\n"
                              f"neighbour purity {PUR[m]['all']:.2f}",
                              fontsize=9.3, color=INK, rotation=0, ha="right",
                              va="center", labelpad=12)

    fig.suptitle("t-SNE of each representation, one EC class highlighted per panel",
                 x=0.135, ha="left", y=1.0, fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.135, 0.955,
             "Same t-SNE settings for all rows (standardize → PCA 50 → perplexity 30, "
             "seed 42). Neighbour purity = share of each sequence's 15 nearest neighbours "
             "(full feature space) in the same class; chance = 0.14.",
             fontsize=9.3, color=INK_2, va="top")
    plt.subplots_adjust(left=0.135, right=0.995, top=0.88, bottom=0.02,
                        wspace=0.04, hspace=0.06)
    out = W / "fig6_tsne_by_class.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
