"""Two extra figures for the paper: per-class F1 drop, identity vs EC level."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/home/claude/paper")
R = json.loads(Path("/home/claude/analysis/bootstrap_results.json").read_text())
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8980", "#ebeae4"
COL = {"B": "#2a78d6", "C": "#eb6834"}
NAME = {"B": "B: ESM-2 8M", "C": "C: ESM-2 650M"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5ce")
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)


# ---- per-class F1 drop ----
fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
style(ax)
classes = [f"EC{c}" for c in range(1, 8)]
names = ["Oxidoreductase", "Transferase", "Hydrolase", "Lyase", "Isomerase",
         "Ligase", "Translocase"]
for m, off in (("B", -0.13), ("C", 0.13)):
    d = R["per_class_f1_drop"][m]
    xs = np.arange(7) + off
    for xx, c in zip(xs, classes):
        ax.plot([xx, xx], d[c]["ci95"], color=COL[m], lw=2.6, solid_capstyle="round")
    ax.scatter(xs, [d[c]["estimate"] for c in classes], s=70, color=COL[m],
               edgecolor=SURFACE, linewidth=1.8, zorder=3, label=NAME[m])
ax.axvspan(3.55, 4.45, color="#f4ece6", zorder=0)
ax.set_xticks(range(7))
ax.set_xticklabels([f"{c}\n{n}" for c, n in zip(classes, names)], fontsize=10, color=INK)
ax.set_ylim(0, 0.24)
ax.set_ylabel("per-class F1 drop  (random − J<0.05)", fontsize=10.5, color=INK_2)
ax.legend(frameon=False, fontsize=10, loc="upper left")
fig.savefig(OUT / "fig_perclass_drop.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ---- identity vs EC level ----
t = pd.read_csv("/home/claude/bio/out/idea2_identity_vs_EC_level.csv")
fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor=SURFACE)
style(ax)
x = np.arange(len(t))
shades = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]       # one hue, light -> dark
for L, col in zip(range(1, 5), shades):
    v = t[f"same_EC_level{L}"].to_numpy()
    ax.plot(x, v, marker="o", color=col, lw=2, ms=7, markeredgecolor=SURFACE,
            markeredgewidth=1.8)
    ax.text(x[0] - 0.12, v[0], f"EC level {L}  {v[0]*100:.0f}%", ha="right", va="center",
            fontsize=9.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"{b}\nn = {n:,}" for b, n in zip(t.identity_bin, t.n_pairs)],
                   fontsize=9.5, color=INK)
ax.set_xlim(-1.75, len(t) - 0.6)
ax.set_ylim(0, 1.04)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("share of pairs with the same EC number\ndown to that level", fontsize=10.5,
              color=INK_2)
ax.set_xlabel("alignment identity of the pair (both sequences ≥80% covered)",
              fontsize=10.5, color=INK_2)
fig.savefig(OUT / "fig_identity_ec.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("ok")
