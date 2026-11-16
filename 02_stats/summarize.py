"""Merge bootstrap + seed results into one table (CSV) and one JSON."""
import csv
import json
from pathlib import Path

import numpy as np

WORK = Path("/home/claude/analysis")
R = json.loads((WORK / "bootstrap_results.json").read_text())
S = json.loads((WORK / "seeds_results.json").read_text())
SPLITS = ["random", "cluster@0.3", "cluster@0.2", "cluster@0.1", "cluster@0.05"]


def seed_stats(fn):
    v = np.array([fn(r) for r in S.values()])
    return float(v.mean()), float(v.std(ddof=1)), float(v.min()), float(v.max())


rows = []
for m in "ABC":
    for s in SPLITS:
        b = R["macro_f1"][f"{m}|{s}"]
        mu, sd, lo, hi = seed_stats(lambda r: r[f"{m}|{s}"])
        rows.append(["macro_F1", m, s, b["estimate"], *b["ci95"], mu, sd, lo, hi, ""])

for m in "ABC":
    b = R["drop_random_minus_strict"][m]
    mu, sd, lo, hi = seed_stats(lambda r: r[f"{m}|random"] - r[f"{m}|cluster@0.05"])
    rows.append(["drop_abs", m, "random-0.05", b["estimate"], *b["ci95"], mu, sd, lo, hi,
                 b["p_bootstrap_two_sided"]])
for m in "ABC":
    b = R["relative_drop"][m]
    mu, sd, lo, hi = seed_stats(
        lambda r: (r[f"{m}|random"] - r[f"{m}|cluster@0.05"]) / r[f"{m}|random"])
    rows.append(["drop_rel", m, "random-0.05", b["estimate"], *b["ci95"], mu, sd, lo, hi,
                 b["p_bootstrap_two_sided"]])

def drop(r, m):
    return r[f"{m}|random"] - r[f"{m}|cluster@0.05"]

def rdrop(r, m):
    return drop(r, m) / r[f"{m}|random"]

for key, label, fn in [
    ("drop_B_minus_drop_C", "dropB-dropC", lambda r: drop(r, "B") - drop(r, "C")),
    ("reldrop_B_minus_reldrop_C", "reldropB-reldropC", lambda r: rdrop(r, "B") - rdrop(r, "C")),
    ("drop_B_minus_drop_A", "dropB-dropA", lambda r: drop(r, "B") - drop(r, "A")),
    ("drop_C_minus_drop_A", "dropC-dropA", lambda r: drop(r, "C") - drop(r, "A")),
    ("strict_gap_C_minus_B", "C-B@0.05", lambda r: r["C|cluster@0.05"] - r["B|cluster@0.05"]),
    ("strict_gap_B_minus_A", "B-A@0.05", lambda r: r["B|cluster@0.05"] - r["A|cluster@0.05"]),
    ("C_strict_minus_B_random", "C@0.05-B@random", lambda r: r["C|cluster@0.05"] - r["B|random"]),
]:
    b = R[key]
    mu, sd, lo, hi = seed_stats(fn)
    rows.append(["comparison", "", label, b["estimate"], *b["ci95"], mu, sd, lo, hi,
                 b["p_bootstrap_two_sided"]])

header = ["quantity", "model", "condition", "estimate_seed42", "boot_ci95_low",
          "boot_ci95_high", "seed_mean", "seed_sd", "seed_min", "seed_max", "p_bootstrap"]
out = WORK / "figs" / "stats_table.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in rows:
        w.writerow([r[0], r[1], r[2]] + [f"{x:.4f}" if isinstance(x, float) else x for x in r[3:]])
print("saved", out)

merged = {"bootstrap": R, "seeds": S}
(WORK / "figs" / "stats_results.json").write_text(json.dumps(merged, indent=1))
print("saved stats_results.json")

for r in rows:
    if r[0] in ("drop_abs", "drop_rel", "comparison"):
        p = r[10]
        ptxt = "" if p == "" else ("p<0.001" if p < 0.001 else f"p={p:.3f}")
        print(f"{r[0]:<11}{r[1]:<2}{r[2]:<18} est {r[3]:+.3f}  CI[{r[4]:+.3f},{r[5]:+.3f}]  "
              f"seeds {r[6]:+.3f}±{r[7]:.3f} (min {r[8]:+.3f}, max {r[9]:+.3f})  {ptxt}")
