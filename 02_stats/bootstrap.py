"""Cluster bootstrap on the seed-42 out-of-fold predictions.

Why clusters and not single sequences:
  members of one sequence family are not independent. Resampling single
  sequences would pretend we have 7,000 independent data points and give
  intervals that are too narrow. We resample whole families instead, using the
  strictest family definition (4-mer Jaccard >= 0.05, 3,580 families), which is
  the most conservative choice.

Implementation:
  drawing families with replacement is equivalent to giving every sequence a
  weight = how many times its family was drawn. A weighted confusion matrix then
  gives macro-F1 exactly, which makes 2,000 replicates take seconds.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

WORK = Path("/home/claude/analysis")
B_REPS = 2000
RNG = np.random.default_rng(2026)

P = np.load(WORK / "preds.npz")
y = P["y"]
SPLITS = [str(s) for s in P["split_names"]]
MODELS = [str(m) for m in P["models"]]
STRICT = "cluster@0.05"
K = 7
clusters = P["clusters_005"]
n_cl = int(clusters.max()) + 1
ytr = y - 1                                        # classes 0..6


def weighted_cm(pred, w):
    return np.bincount(ytr * K + (pred - 1), weights=w,
                       minlength=K * K).reshape(K, K)


def per_class_f1(cm):
    tp = np.diag(cm)
    denom = cm.sum(axis=1) + cm.sum(axis=0)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


def summarize(samples, point):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p_two = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return {"estimate": float(point), "ci95": [float(lo), float(hi)],
            "p_bootstrap_two_sided": float(min(1.0, p_two)),
            "boot_sd": float(samples.std(ddof=1))}


def main():
    preds = {(m, s): P[f"pred__{m}__{s}"] for m in MODELS for s in SPLITS}

    # sanity: weight 1 everywhere must reproduce sklearn exactly
    w1 = np.ones(len(y))
    for (m, s), p in preds.items():
        mine = per_class_f1(weighted_cm(p, w1)).mean()
        ref = f1_score(y, p, average="macro")
        assert abs(mine - ref) < 1e-12, (m, s, mine, ref)
    print("weighted macro-F1 reproduces sklearn exactly")

    point_f1 = {k: per_class_f1(weighted_cm(p, w1)).mean() for k, p in preds.items()}
    point_cls = {k: per_class_f1(weighted_cm(p, w1)) for k, p in preds.items()}

    boot_f1 = {k: np.empty(B_REPS) for k in preds}
    boot_cls = {k: np.empty((B_REPS, K)) for k in preds}
    for b in range(B_REPS):
        draw = RNG.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl)[clusters].astype(float)
        for k, p in preds.items():
            f = per_class_f1(weighted_cm(p, w))
            boot_cls[k][b] = f
            boot_f1[k][b] = f.mean()

    res = {"n_bootstrap": B_REPS,
           "resampling_unit": "sequence family (4-mer Jaccard >= 0.05, connected components)",
           "n_families": n_cl}

    # 1) macro-F1 with CI, every model x split
    res["macro_f1"] = {f"{m}|{s}": summarize(boot_f1[(m, s)], point_f1[(m, s)])
                       for m in MODELS for s in SPLITS}

    # 2) drop random -> strict, per model
    drop = {m: boot_f1[(m, "random")] - boot_f1[(m, STRICT)] for m in MODELS}
    drop_pt = {m: point_f1[(m, "random")] - point_f1[(m, STRICT)] for m in MODELS}
    res["drop_random_minus_strict"] = {m: summarize(drop[m], drop_pt[m]) for m in MODELS}

    rel = {m: drop[m] / boot_f1[(m, "random")] for m in MODELS}
    rel_pt = {m: drop_pt[m] / point_f1[(m, "random")] for m in MODELS}
    res["relative_drop"] = {m: summarize(rel[m], rel_pt[m]) for m in MODELS}

    # 3) the key comparison: is B's drop different from C's drop?
    res["drop_B_minus_drop_C"] = summarize(drop["B"] - drop["C"], drop_pt["B"] - drop_pt["C"])
    res["reldrop_B_minus_reldrop_C"] = summarize(rel["B"] - rel["C"], rel_pt["B"] - rel_pt["C"])
    res["drop_B_minus_drop_A"] = summarize(drop["B"] - drop["A"], drop_pt["B"] - drop_pt["A"])
    res["drop_C_minus_drop_A"] = summarize(drop["C"] - drop["A"], drop_pt["C"] - drop_pt["A"])

    # 4) gaps between models at the strict split
    for a, b in [("B", "A"), ("C", "B"), ("C", "A")]:
        res[f"strict_gap_{a}_minus_{b}"] = summarize(
            boot_f1[(a, STRICT)] - boot_f1[(b, STRICT)],
            point_f1[(a, STRICT)] - point_f1[(b, STRICT)])
    # C strict vs B random (the "big model without leakage beats small model with it" line)
    res["C_strict_minus_B_random"] = summarize(
        boot_f1[("C", STRICT)] - boot_f1[("B", "random")],
        point_f1[("C", STRICT)] - point_f1[("B", "random")])

    # 5) per-class F1 drop, and is EC5 the outlier?
    res["per_class_f1_drop"] = {}
    res["ec5_drop_minus_mean_other_drop"] = {}
    for m in MODELS:
        d = boot_cls[(m, "random")] - boot_cls[(m, STRICT)]
        d_pt = point_cls[(m, "random")] - point_cls[(m, STRICT)]
        res["per_class_f1_drop"][m] = {
            f"EC{c+1}": summarize(d[:, c], d_pt[c]) for c in range(K)}
        others = [c for c in range(K) if c != 4]
        res["ec5_drop_minus_mean_other_drop"][m] = summarize(
            d[:, 4] - d[:, others].mean(axis=1), d_pt[4] - d_pt[others].mean())

    (WORK / "bootstrap_results.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(WORK / "bootstrap_samples.npz",
                        **{f"f1__{m}__{s}": boot_f1[(m, s)] for m in MODELS for s in SPLITS})
    print("saved bootstrap_results.json")

    # ---- console summary ----
    def show(label, r):
        lo, hi = r["ci95"]
        print(f"  {label:<38} {r['estimate']:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
              f"p = {r['p_bootstrap_two_sided']:.4f}")

    print("\nmacro-F1 (95% CI)")
    for m in MODELS:
        row = "  " + m + "  " + "  ".join(
            f"{res['macro_f1'][f'{m}|{s}']['estimate']:.3f}"
            f"[{res['macro_f1'][f'{m}|{s}']['ci95'][0]:.3f},{res['macro_f1'][f'{m}|{s}']['ci95'][1]:.3f}]"
            for s in SPLITS)
        print(row)
    print("\ndrop, random -> strict")
    for m in MODELS:
        show(f"{m} absolute drop", res["drop_random_minus_strict"][m])
    for m in MODELS:
        show(f"{m} relative drop", res["relative_drop"][m])
    print("\nkey comparisons")
    show("drop B - drop C (absolute)", res["drop_B_minus_drop_C"])
    show("drop B - drop C (relative)", res["reldrop_B_minus_reldrop_C"])
    show("drop B - drop A", res["drop_B_minus_drop_A"])
    show("drop C - drop A", res["drop_C_minus_drop_A"])
    show("strict: B - A", res["strict_gap_B_minus_A"])
    show("strict: C - B", res["strict_gap_C_minus_B"])
    show("C strict - B random", res["C_strict_minus_B_random"])
    print("\nper-class F1 drop")
    for m in ("B", "C"):
        for c in range(K):
            show(f"{m} EC{c+1}", res["per_class_f1_drop"][m][f"EC{c+1}"])
        show(f"{m} EC5 - mean(other classes)", res["ec5_drop_minus_mean_other_drop"][m])


if __name__ == "__main__":
    main()
