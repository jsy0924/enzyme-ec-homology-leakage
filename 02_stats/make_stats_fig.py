"""Figure 4: the main result with uncertainty.

A. macro-F1 across splits with 95% cluster-bootstrap CIs (3 models)
B. drop random -> strict per model: bootstrap CI + each of 10 fold seeds
C. per-class F1 drop for ESM-2 8M and 650M (EC5 is the outlier)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK = Path("/home/claude/analysis")
R = json.loads((WORK / "bootstrap_results.json").read_text())
SEEDS = json.loads((WORK / "seeds_results.json").read_text())

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8980"
GRID = "#ebeae4"
COL = {"A": "#8a8980", "B": "#2a78d6", "C": "#eb6834"}
NAME = {"A": "A: composition", "B": "B: ESM-2 8M", "C": "C: ESM-2 650M"}
SPLITS = ["random", "cluster@0.3", "cluster@0.2", "cluster@0.1", "cluster@0.05"]
XLAB = ["random\n(leakage)", "J<0.30", "J<0.20", "J<0.10", "J<0.05"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5ce")
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)


def main():
    fig = plt.figure(figsize=(14.5, 10.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 0.92],
                          hspace=0.46, wspace=0.22,
                          left=0.07, right=0.97, top=0.87, bottom=0.07)

    # ---------- A ----------
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    x = np.arange(len(SPLITS))
    for m in "ABC":
        est = np.array([R["macro_f1"][f"{m}|{s}"]["estimate"] for s in SPLITS])
        lo = np.array([R["macro_f1"][f"{m}|{s}"]["ci95"][0] for s in SPLITS])
        hi = np.array([R["macro_f1"][f"{m}|{s}"]["ci95"][1] for s in SPLITS])
        ax.fill_between(x, lo, hi, color=COL[m], alpha=0.16, lw=0)
        ax.plot(x, est, "-o", color=COL[m], lw=2, ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=NAME[m])
        ax.text(x[-1] + 0.12, est[-1], f"{NAME[m]}\n{est[-1]:.3f}",
                va="center", fontsize=9.2, color=INK)
    ax.axhline(1 / 7, ls="--", color="#c9c8c0", lw=1.1)
    ax.text(0, 1 / 7 + 0.012, "chance (1/7)", fontsize=8.8, color=INK_3)
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB, fontsize=9.5, color=INK)
    ax.set_xlim(-0.3, len(SPLITS) - 0.3 + 0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("macro-F1", fontsize=10.5, color=INK_2)
    ax.set_xlabel("split   (looser  →  stricter: less homology leakage)",
                  fontsize=10, color=INK_2)
    ax.set_title("A.  macro-F1 with 95% cluster-bootstrap CI",
                 loc="left", fontsize=12.2, fontweight="bold", color=INK, pad=10)

    # ---------- B ----------
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    rng = np.random.default_rng(0)
    for i, m in enumerate("ABC"):
        r = R["drop_random_minus_strict"][m]
        lo, hi = r["ci95"]
        seed_drops = [v[f"{m}|random"] - v[f"{m}|cluster@0.05"] for v in SEEDS.values()]
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(seed_drops)), seed_drops,
                   s=16, color=COL[m], alpha=0.45, lw=0, zorder=2)
        ax.plot([i, i], [lo, hi], color=COL[m], lw=3.2, solid_capstyle="round", zorder=3)
        ax.scatter([i], [r["estimate"]], s=95, color=COL[m],
                   edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(i + 0.2, r["estimate"], f"{r['estimate']:.3f}",
                va="center", fontsize=9.6, fontweight="bold", color=INK)
    kb = R["drop_B_minus_drop_C"]
    yb = max(R["drop_random_minus_strict"][m]["ci95"][1] for m in "BC") + 0.012
    ax.plot([1, 1, 2, 2], [yb - 0.004, yb, yb, yb - 0.004], color=INK_2, lw=1.1)
    ax.text(1.5, yb + 0.004,
            f"B vs C: Δ = {kb['estimate']:+.3f}, p = {kb['p_bootstrap_two_sided']:.2f}  (n.s.)",
            ha="center", va="bottom", fontsize=9.2, color=INK_2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAME[m].replace(": ", ":\n") for m in "ABC"], fontsize=9.5, color=INK)
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(0, yb + 0.03)
    ax.set_ylabel("macro-F1 drop  (random − J<0.05)", fontsize=10, color=INK_2)
    ax.set_title("B.  How much each model loses", loc="left",
                 fontsize=12.2, fontweight="bold", color=INK, pad=10)
    ax.text(0.0, -0.215, "line = 95% bootstrap CI   small dots = 10 fold-assignment seeds",
            transform=ax.transAxes, fontsize=8.6, color=INK_3)

    # ---------- C ----------
    ax = fig.add_subplot(gs[1, :])
    style(ax)
    classes = [f"EC{c}" for c in range(1, 8)]
    off = {"B": -0.13, "C": 0.13}
    for m in "BC":
        d = R["per_class_f1_drop"][m]
        est = [d[c]["estimate"] for c in classes]
        lo = [d[c]["ci95"][0] for c in classes]
        hi = [d[c]["ci95"][1] for c in classes]
        xs = np.arange(7) + off[m]
        for xx, l, h in zip(xs, lo, hi):
            ax.plot([xx, xx], [l, h], color=COL[m], lw=2.6, solid_capstyle="round")
        ax.scatter(xs, est, s=70, color=COL[m], edgecolor=SURFACE, linewidth=1.8,
                   zorder=3, label=NAME[m])
    ax.axvspan(3.55, 4.45, color="#f4ece6", zorder=0)
    names = ["Oxidoreductase", "Transferase", "Hydrolase", "Lyase",
             "Isomerase", "Ligase", "Translocase"]
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"{c}\n{n}" for c, n in zip(classes, names)],
                       fontsize=9.5, color=INK)
    ax.set_ylim(0, 0.29)
    ax.set_ylabel("per-class F1 drop  (random − J<0.05)", fontsize=10, color=INK_2)
    e5b = R["ec5_drop_minus_mean_other_drop"]["B"]
    e5c = R["ec5_drop_minus_mean_other_drop"]["C"]
    ax.text(4, 0.284,
            f"EC5 drops more than the other classes on average\n"
            f"8M: +{e5b['estimate']:.3f} [{e5b['ci95'][0]:+.3f}, {e5b['ci95'][1]:+.3f}]    "
            f"650M: +{e5c['estimate']:.3f} [{e5c['ci95'][0]:+.3f}, {e5c['ci95'][1]:+.3f}]",
            ha="center", va="top", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("C.  Which enzyme classes depend most on homology", loc="left",
                 fontsize=12.2, fontweight="bold", color=INK, pad=10)

    fig.suptitle("Homology leakage: effect sizes with uncertainty",
                 x=0.07, ha="left", y=0.975, fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.07, 0.938,
             f"n = 7,000 enzymes  |  CIs from {R['n_bootstrap']:,} bootstrap resamples of "
             f"{R['n_families']:,} sequence families (not single sequences)  |  "
             f"seed spread from {len(SEEDS)} fold assignments",
             fontsize=9.6, color=INK_2, va="top")

    out = WORK / "figs" / "fig4_statistics.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
