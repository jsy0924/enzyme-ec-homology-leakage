"""Figure 5: adding BLAST (D) and the hybrid (D+C).

A. macro-F1 across splits with 95% CIs, five predictors
B. drop random -> strict: CI + 10 seed dots
C. accuracy by identity of the closest training sequence (5 splits pooled)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = Path("/home/claude/blast")
A_DIR = Path("/home/claude/analysis")
R = json.loads((W / "bootstrap_all.json").read_text())
ID = json.loads((W / "analyze_D.json").read_text())["identity_bins"]
S1 = json.loads((A_DIR / "seeds_results.json").read_text())      # A, B, C
S2 = json.loads((W / "seeds_D_results.json").read_text())        # C, D, D+C

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8980", "#ebeae4"
COL = {"A": "#8a8980", "B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a", "D+C": "#3d3c38"}
DASH = {"D+C": (0, (4, 2.2))}
NAME = {"A": "A: composition", "B": "B: ESM-2 8M", "C": "C: ESM-2 650M",
        "D": "D: BLAST (DIAMOND)", "D+C": "D+C: BLAST, else 650M"}
SPLITS = ["random", "cluster@0.3", "cluster@0.2", "cluster@0.1", "cluster@0.05"]
XLAB = ["random\n(leakage)", "J<0.30", "J<0.20", "J<0.10", "J<0.05"]
ORDER = ["A", "B", "C", "D", "D+C"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5ce")
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)


def seed_drops(m):
    src = S2 if m in ("D", "D+C") else S1
    return [v[f"{m}|random"] - v[f"{m}|cluster@0.05"] for v in src.values()]


def main():
    fig = plt.figure(figsize=(15, 10.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 0.95],
                          hspace=0.5, wspace=0.24,
                          left=0.065, right=0.97, top=0.875, bottom=0.075)

    # ---------- A ----------
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    x = np.arange(len(SPLITS))
    ends = {}
    for m in ORDER:
        est = np.array([R["macro_f1"][f"{m}|{s}"]["estimate"] for s in SPLITS])
        lo = np.array([R["macro_f1"][f"{m}|{s}"]["ci95"][0] for s in SPLITS])
        hi = np.array([R["macro_f1"][f"{m}|{s}"]["ci95"][1] for s in SPLITS])
        if m != "D+C":
            ax.fill_between(x, lo, hi, color=COL[m], alpha=0.14, lw=0)
        ax.plot(x, est, marker="o", color=COL[m], lw=2, ms=7, linestyle=DASH.get(m, "-"),
                markeredgecolor=SURFACE, markeredgewidth=1.8)
        ends[m] = est[-1]
    # direct labels at the right end, nudged apart where lines crowd together
    ypos = {"A": ends["A"], "B": ends["B"], "D": ends["D"] - 0.028,
            "C": ends["C"] + 0.004, "D+C": ends["D+C"] + 0.034}
    for m in ORDER:
        ax.annotate(f"{NAME[m]}  {ends[m]:.3f}", xy=(x[-1], ends[m]),
                    xytext=(x[-1] + 0.14, ypos[m]), va="center", fontsize=9,
                    color=INK, arrowprops=dict(arrowstyle="-", color="#c9c8c0", lw=0.8))
    ax.axhline(1 / 7, ls="--", color="#c9c8c0", lw=1.1)
    ax.text(0, 1 / 7 + 0.012, "chance (1/7)", fontsize=8.8, color=INK_3)
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB, fontsize=9.5, color=INK)
    ax.set_xlim(-0.3, len(SPLITS) + 1.05)
    ax.set_ylim(0, 1)
    ax.set_ylabel("macro-F1", fontsize=10.5, color=INK_2)
    ax.set_xlabel("split   (looser  →  stricter)", fontsize=10, color=INK_2)
    ax.set_title("A.  macro-F1 with 95% cluster-bootstrap CI", loc="left",
                 fontsize=12.2, fontweight="bold", color=INK, pad=10)

    # ---------- B ----------
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    rng = np.random.default_rng(0)
    for i, m in enumerate(ORDER):
        r = R["drop"][m]
        sd = seed_drops(m)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(sd)), sd, s=14,
                   color=COL[m], alpha=0.45, lw=0, zorder=2)
        ax.plot([i, i], r["ci95"], color=COL[m], lw=3, solid_capstyle="round", zorder=3)
        ax.scatter([i], [r["estimate"]], s=85, color=COL[m],
                   edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(i + 0.19, r["estimate"], f"{r['estimate']:.3f}", va="center",
                fontsize=9.2, fontweight="bold", color=INK)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(["A\ncomp.", "B\n8M", "C\n650M", "D\nBLAST", "D+C\nhybrid"],
                       fontsize=9.5, color=INK)
    ax.set_xlim(-0.5, len(ORDER) - 0.3)
    ax.set_ylim(0, 0.145)
    ax.set_ylabel("macro-F1 drop  (random − J<0.05)", fontsize=10, color=INK_2)
    ax.set_title("B.  How much each predictor loses", loc="left",
                 fontsize=12.2, fontweight="bold", color=INK, pad=10)
    c = R["comparisons"]
    ax.text(0.0, -0.25,
            f"B, C, D drops do not differ (B−C p={c['drop B - drop C']['p_bootstrap_two_sided']:.2f}, "
            f"D−C p={c['drop D - drop C']['p_bootstrap_two_sided']:.2f}, "
            f"D−B p={c['drop D - drop B']['p_bootstrap_two_sided']:.2f})\n"
            "line = 95% bootstrap CI   small dots = 10 fold seeds",
            transform=ax.transAxes, fontsize=8.6, color=INK_3, va="top")

    # ---------- C ----------
    ax = fig.add_subplot(gs[1, :])
    style(ax)
    bins = [">=90%", "70-90%", "50-70%", "40-50%", "30-40%", "<30%", "no hit"]
    labels = ["≥90%", "70–90%", "50–70%", "40–50%",
              "30–40%", "<30%", "no hit\n(BLAST finds\nnothing)"]
    xb = np.arange(len(bins))
    ax.axvspan(len(bins) - 1.5, len(bins) - 0.5, color="#f4ece6", zorder=0)
    end_note = {
        "A": "A: composition",
        "B": "B: ESM-2 8M",
        "C": "C: ESM-2 650M — still right where BLAST finds nothing",
        "D": "D: BLAST — no hit, no answer",
    }
    for m in ["A", "B", "C", "D"]:
        v = [ID[b][f"acc_{m}"] for b in bins]
        ax.plot(xb, v, marker="o", color=COL[m], lw=2, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.8)
        ax.text(xb[-1] + 0.14, v[-1], f"{end_note[m]}  ({v[-1]*100:.0f}%)",
                va="center", fontsize=9.2, color=INK)
    ax.set_xticks(xb)
    ax.set_xticklabels([f"{lab}\nn = {ID[b]['n_pooled']:,}" for lab, b in
                        zip([l.split(chr(10))[0] for l in labels], bins)],
                       fontsize=9.3, color=INK)
    ax.set_xlim(-0.4, len(bins) + 2.3)
    ax.set_ylim(-0.02, 1.04)
    ax.set_ylabel("accuracy", fontsize=10.5, color=INK_2)
    ax.set_xlabel("identity of the closest training sequence  (high  →  none;  "
                  "shaded = BLAST finds no hit)", fontsize=10, color=INK_2, labelpad=8)
    ax.set_title("C.  Accuracy by how similar the nearest training sequence really is "
                 "(all 5 splits pooled)", loc="left", fontsize=12.2,
                 fontweight="bold", color=INK, pad=10)

    fig.suptitle("Adding BLAST: same loss, different failure mode",
                 x=0.065, ha="left", y=0.975, fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.065, 0.938,
             "D = best-bit-score DIAMOND blastp hit among training sequences "
             "(e ≤ 1e-3); no hit counts as wrong  |  D+C uses BLAST when it has a hit, "
             "ESM-2 650M otherwise", fontsize=9.6, color=INK_2, va="top")

    out = W / "fig5_blast.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
