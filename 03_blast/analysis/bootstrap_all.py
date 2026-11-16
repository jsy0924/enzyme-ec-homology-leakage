"""Cluster bootstrap for A, B, C, D (BLAST) and D+C (hybrid), seed-42 folds.

Same method as before: resample the 3,580 sequence families with replacement.
D can output 0 ("no hit"): it counts as a miss for the true class and as a
prediction for no class, so the confusion matrix has an extra column.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ANALYSIS = Path("/home/claude/analysis")
WORK = Path("/home/claude/blast")
B_REPS = 2000
RNG = np.random.default_rng(2026)
K = 7
STRICT = "cluster@0.05"

P = np.load(ANALYSIS / "preds.npz")
D = np.load(WORK / "preds_D.npz")
y = P["y"]
SPLITS = [str(s) for s in P["split_names"]]
clusters = P["clusters_005"]
n_cl = int(clusters.max()) + 1
MODELS = ["A", "B", "C", "D", "D+C"]

preds = {}
for s in SPLITS:
    for m in "ABC":
        preds[(m, s)] = P[f"pred__{m}__{s}"]
    preds[("D", s)] = D[f"pred__D__{s}"]
    preds[("D+C", s)] = np.where(preds[("D", s)] > 0, preds[("D", s)], preds[("C", s)])


def weighted_cm(pred, w):
    # rows: true 1..7, cols: predicted 0..7 (0 = no prediction)
    return np.bincount((y - 1) * (K + 1) + pred, weights=w,
                       minlength=K * (K + 1)).reshape(K, K + 1)


def per_class_f1(cm):
    tp = np.array([cm[c, c + 1] for c in range(K)])
    denom = cm.sum(axis=1) + cm[:, 1:].sum(axis=0)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


def summarize(samples, point):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return {"estimate": float(point), "ci95": [float(lo), float(hi)],
            "p_bootstrap_two_sided": float(min(1.0, p))}


def main():
    w1 = np.ones(len(y))
    for k, p in preds.items():
        mine = per_class_f1(weighted_cm(p, w1)).mean()
        ref = f1_score(y, p, labels=list(range(1, 8)), average="macro", zero_division=0)
        assert abs(mine - ref) < 1e-12, (k, mine, ref)
    print("weighted macro-F1 (with no-hit column) reproduces sklearn exactly")

    pt = {k: per_class_f1(weighted_cm(p, w1)).mean() for k, p in preds.items()}
    bs = {k: np.empty(B_REPS) for k in preds}
    for b in range(B_REPS):
        draw = RNG.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl)[clusters].astype(float)
        for k, p in preds.items():
            bs[k][b] = per_class_f1(weighted_cm(p, w)).mean()

    res = {"n_bootstrap": B_REPS, "n_families": n_cl,
           "macro_f1": {f"{m}|{s}": summarize(bs[(m, s)], pt[(m, s)])
                        for m in MODELS for s in SPLITS}}
    drop = {m: bs[(m, "random")] - bs[(m, STRICT)] for m in MODELS}
    drop_pt = {m: pt[(m, "random")] - pt[(m, STRICT)] for m in MODELS}
    res["drop"] = {m: summarize(drop[m], drop_pt[m]) for m in MODELS}

    comps = {
        "drop D - drop C": (drop["D"] - drop["C"], drop_pt["D"] - drop_pt["C"]),
        "drop D - drop B": (drop["D"] - drop["B"], drop_pt["D"] - drop_pt["B"]),
        "drop D - drop A": (drop["D"] - drop["A"], drop_pt["D"] - drop_pt["A"]),
        "drop B - drop C": (drop["B"] - drop["C"], drop_pt["B"] - drop_pt["C"]),
        "drop C - drop D+C": (drop["C"] - drop["D+C"], drop_pt["C"] - drop_pt["D+C"]),
        "drop D - drop D+C": (drop["D"] - drop["D+C"], drop_pt["D"] - drop_pt["D+C"]),
    }
    for s, tag in [("random", "random"), (STRICT, "strict")]:
        for a, b in [("C", "D"), ("D", "B"), ("D+C", "C"), ("D+C", "D")]:
            comps[f"{tag}: {a} - {b}"] = (bs[(a, s)] - bs[(b, s)], pt[(a, s)] - pt[(b, s)])
    res["comparisons"] = {k: summarize(*v) for k, v in comps.items()}

    (WORK / "bootstrap_all.json").write_text(json.dumps(res, indent=1))
    print("saved bootstrap_all.json\n")

    print("macro-F1 [95% CI]")
    for m in MODELS:
        print(f"  {m:<4}" + "  ".join(
            f"{res['macro_f1'][f'{m}|{s}']['estimate']:.3f}"
            f"[{res['macro_f1'][f'{m}|{s}']['ci95'][0]:.3f},{res['macro_f1'][f'{m}|{s}']['ci95'][1]:.3f}]"
            for s in SPLITS))
    print("\ndrop random -> strict")
    for m in MODELS:
        r = res["drop"][m]
        print(f"  {m:<4} {r['estimate']:+.3f}  CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    print("\ncomparisons")
    for k, r in res["comparisons"].items():
        p = r["p_bootstrap_two_sided"]
        print(f"  {k:<20} {r['estimate']:+.3f}  CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
              f"{'p<0.001' if p < 0.001 else f'p={p:.3f}'}")


if __name__ == "__main__":
    main()
