"""Figure: what the amino-acid composition baseline (Model A) actually learned.

Panel A: logistic-regression coefficients, 7 EC classes x 20 amino acids.
Panel B: the EC7 (translocase) row on its own, the clearest signal.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CSV = "/mnt/user-data/uploads/Documents/enzyme-project/sequences_main.csv"
OUT = "/home/claude/fig3_composition_weights.png"

# ---- design tokens (reference palette) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8980"
POLE_NEG = "#2a78d6"     # blue
MIDPOINT = "#f0efec"     # neutral gray
POLE_POS = "#e34948"     # red

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", [POLE_NEG, MIDPOINT, POLE_POS], N=256)

AA = "ACDEFGHIKLMNPQRSTVWY"
EC_NAMES = {1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase",
            4: "Lyase", 5: "Isomerase", 6: "Ligase", 7: "Translocase"}

# amino acids grouped by chemical property -> the pattern becomes readable
GROUPS = [
    ("Hydrophobic", list("AVILMFWY")),
    ("Polar",       list("STNQCGP")),
    ("Positive",    list("KRH")),
    ("Negative",    list("DE")),
]
ORDER = [a for _, g in GROUPS for a in g]
ORDER_IDX = [AA.index(a) for a in ORDER]


def composition(seq):
    counts = np.zeros(20)
    n = 0
    for ch in seq:
        i = AA.find(ch)
        if i >= 0:
            counts[i] += 1
            n += 1
    return counts / n if n else counts


def main():
    df = pd.read_csv(CSV)
    X = np.vstack([composition(s) for s in df["sequence"]])
    y = df["label"].to_numpy()
    classes = sorted(set(y.tolist()))

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
    clf.fit(X, y)
    W = clf[-1].coef_[:, ORDER_IDX]          # (7, 20) reordered

    lim = float(np.abs(W).max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig = plt.figure(figsize=(13.2, 8.1), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.62,
                          left=0.115, right=0.9, top=0.835, bottom=0.08)

    # ---------------- Panel A : heatmap ----------------
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURFACE)
    im = ax.imshow(W, cmap=DIVERGING, norm=norm, aspect="auto",
                   interpolation="nearest")

    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=10.5, color=INK)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"EC{c}  {EC_NAMES[c]}" for c in classes],
                       fontsize=10.5, color=INK)
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)

    # 2px surface gap between cells
    ax.set_xticks(np.arange(-.5, len(ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(classes), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    # selective direct labels: only the cells that carry the story
    thr = 0.30
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if abs(W[i, j]) >= thr:
                ax.text(j, i, f"{W[i,j]:+.2f}", ha="center", va="center",
                        fontsize=8.4, fontweight="bold",
                        color="#ffffff" if abs(W[i, j]) > lim * 0.62 else INK)

    # property-group brackets under the x axis
    # property-group brackets BELOW the residue letters
    pos = 0
    for name, members in GROUPS:
        start, end = pos - 0.5, pos + len(members) - 0.5
        ax.plot([start + 0.08, end - 0.08], [7.22, 7.22], color=INK_MUTED,
                lw=1.3, clip_on=False)
        ax.text((start + end) / 2, 7.40, name, ha="center", va="top",
                fontsize=9.8, color=INK_2, clip_on=False)
        if pos:
            ax.axvline(start, color=SURFACE, lw=3.5)
        pos += len(members)

    ax.set_title("A.  Learned weight of each amino acid, per enzyme class",
                 fontsize=12.6, fontweight="bold", color=INK,
                 loc="left", pad=13)

    cb = fig.colorbar(im, ax=ax, fraction=0.021, pad=0.014)
    cb.ax.tick_params(labelsize=9, length=0, colors=INK_2)
    cb.outline.set_visible(False)
    cb.set_label("coefficient   (blue: less of it  →  red: more of it)",
                 fontsize=9.2, color=INK_2)

    # ---------------- Panel B : EC7 row ----------------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(SURFACE)
    row = W[classes.index(7)]
    colors = [POLE_POS if v > 0 else POLE_NEG for v in row]
    bars = ax2.bar(range(len(ORDER)), row, color=colors, width=0.68)
    for b in bars:                       # 4px rounded data-ends
        b.set_linewidth(0)

    ax2.axhline(0, color=INK_MUTED, lw=1)
    ax2.set_xticks(range(len(ORDER)))
    ax2.set_xticklabels(ORDER, fontsize=10.5, color=INK)
    ax2.set_ylabel("coefficient", fontsize=10, color=INK_2)
    ax2.tick_params(length=0, labelsize=9.5, colors=INK_2)
    for side in ("top", "right", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.spines["left"].set_color("#dcdbd5")
    ax2.grid(axis="y", color="#ebeae4", lw=0.9)
    ax2.set_axisbelow(True)

    pos = 0
    for name, members in GROUPS:
        if pos:
            ax2.axvline(pos - 0.5, color="#dcdbd5", lw=1, ls=":")
        pos += len(members)

    for j, v in enumerate(row):
        if abs(v) >= 0.30:
            ax2.text(j, v + (0.055 if v > 0 else -0.055), f"{v:+.2f}",
                     ha="center", va="bottom" if v > 0 else "top",
                     fontsize=8.8, fontweight="bold", color=INK)

    ax2.set_title("B.  EC7 translocase: hydrophobic residues up, charged residues down",
                  fontsize=12.6, fontweight="bold", color=INK, loc="left", pad=9)
    ax2.set_ylim(min(row) - 0.26, max(row) + 0.26)

    fig.suptitle("What the amino-acid composition baseline learned",
                 fontsize=15.5, fontweight="bold", color=INK,
                 x=0.115, ha="left", y=0.975)
    fig.text(0.115, 0.938,
             "Multinomial logistic regression on 20 residue frequencies  |  "
             "n = 7,000 Swiss-Prot enzymes  |  labels shown for |coef| ≥ 0.30",
             fontsize=9.8, color=INK_2, ha="left", va="top")

    fig.savefig(OUT, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("saved:", OUT)
    print(f"weight range: {-lim:+.2f} .. {lim:+.2f}")


if __name__ == "__main__":
    main()
