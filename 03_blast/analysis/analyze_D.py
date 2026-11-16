"""What does BLAST's result mean? Three checks on the seed-42 predictions.

1. Complementarity: on sequences where BLAST finds nothing in training,
   how do the ESM-2 models do?
2. Identity bins: performance by how similar the closest training sequence
   really is (alignment identity), across all five splits pooled.
3. Hybrid: BLAST when it has a hit, otherwise ESM-2 650M.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ANALYSIS = Path("/home/claude/analysis")
WORK = Path("/home/claude/blast")
P = np.load(ANALYSIS / "preds.npz")
D = np.load(WORK / "preds_D.npz")
y = P["y"]
SPLITS = [str(s) for s in P["split_names"]]
LAB = list(range(1, 8))


def mf1(yt, yp):
    return f1_score(yt, yp, labels=LAB, average="macro", zero_division=0)


res = {"complementarity": {}, "identity_bins": {}, "hybrid": {}}

print("=== 1. where BLAST has no hit in training ===")
for s in SPLITS:
    d = D[f"pred__D__{s}"]
    nohit = d == 0
    row = {"n_nohit": int(nohit.sum()), "frac_nohit": float(nohit.mean())}
    for m in "ABC":
        p = P[f"pred__{m}__{s}"]
        row[f"acc_{m}_on_nohit"] = float((p[nohit] == y[nohit]).mean())
        row[f"acc_{m}_on_hit"] = float((p[~nohit] == y[~nohit]).mean())
    row["acc_D_on_hit"] = float((d[~nohit] == y[~nohit]).mean())
    res["complementarity"][s] = row
    print(f"  {s:<13} no-hit {nohit.sum():4d} ({nohit.mean()*100:4.1f}%)  "
          f"accuracy on no-hit:  A {row['acc_A_on_nohit']:.3f}  "
          f"B {row['acc_B_on_nohit']:.3f}  C {row['acc_C_on_nohit']:.3f}  D 0.000   | "
          f"on hit: C {row['acc_C_on_hit']:.3f}  D {row['acc_D_on_hit']:.3f}")

print("\n=== 2. accuracy by identity of the closest training sequence (5 splits pooled) ===")
BINS = [("no hit", None, None), ("<30%", 0, 30), ("30-40%", 30, 40),
        ("40-50%", 40, 50), ("50-70%", 50, 70), ("70-90%", 70, 90),
        (">=90%", 90, 101)]
yy = np.concatenate([y] * len(SPLITS))
ident = np.concatenate([D[f"ident__D__{s}"] for s in SPLITS])
pp = {m: np.concatenate([P[f"pred__{m}__{s}"] for s in SPLITS]) for m in "ABC"}
pp["D"] = np.concatenate([D[f"pred__D__{s}"] for s in SPLITS])
strict_ident = D["ident__D__cluster@0.05"]
for name, lo, hi in BINS:
    if lo is None:
        mask = np.isnan(ident)
        smask = np.isnan(strict_ident)
    else:
        mask = (ident >= lo) & (ident < hi)
        smask = (strict_ident >= lo) & (strict_ident < hi)
    row = {"n_pooled": int(mask.sum()), "n_strict_split": int(smask.sum())}
    for m in "ABCD":
        row[f"acc_{m}"] = float((pp[m][mask] == yy[mask]).mean()) if mask.any() else None
    res["identity_bins"][name] = row
    print(f"  {name:<8} n={mask.sum():5d} (strict split {smask.sum():4d})   "
          + "   ".join(f"{m} {row[f'acc_{m}']:.3f}" for m in "ABCD"))

print("\n=== 3. hybrid: BLAST if it has a hit, else ESM-2 ===")
for s in SPLITS:
    d = D[f"pred__D__{s}"]
    row = {}
    for m in ("B", "C"):
        h = np.where(d > 0, d, P[f"pred__{m}__{s}"])
        row[f"D+{m}"] = float(mf1(y, h))
        res["hybrid"].setdefault(f"D+{m}", {})[s] = row[f"D+{m}"]
    for m in "BCD":
        p = D[f"pred__D__{s}"] if m == "D" else P[f"pred__{m}__{s}"]
        row[m] = float(mf1(y, p))
    print(f"  {s:<13} B {row['B']:.3f}  C {row['C']:.3f}  D {row['D']:.3f}  "
          f"| D+B {row['D+B']:.3f}  D+C {row['D+C']:.3f}")

(WORK / "analyze_D.json").write_text(json.dumps(res, indent=1))
print("\nsaved analyze_D.json")
