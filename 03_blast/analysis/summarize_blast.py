"""One table + one JSON for everything BLAST-related."""
import csv
import json
from pathlib import Path

import numpy as np

W = Path("/home/claude/blast")
A_DIR = Path("/home/claude/analysis")
R = json.loads((W / "bootstrap_all.json").read_text())
AN = json.loads((W / "analyze_D.json").read_text())
S1 = json.loads((A_DIR / "seeds_results.json").read_text())
S2 = json.loads((W / "seeds_D_results.json").read_text())
SPLITS = ["random", "cluster@0.3", "cluster@0.2", "cluster@0.1", "cluster@0.05"]
SEEDS = list(S2)


def seedvals(m, s):
    src = S2 if m in ("D", "D+C") else S1
    return np.array([src[k][f"{m}|{s}"] for k in SEEDS])


rows = []
for m in ["A", "B", "C", "D", "D+C"]:
    for s in SPLITS:
        b = R["macro_f1"][f"{m}|{s}"]
        v = seedvals(m, s)
        rows.append(["macro_F1", m, s, b["estimate"], *b["ci95"], v.mean(), v.std(ddof=1), ""])
for m in ["A", "B", "C", "D", "D+C"]:
    b = R["drop"][m]
    v = seedvals(m, "random") - seedvals(m, "cluster@0.05")
    rows.append(["drop", m, "random-0.05", b["estimate"], *b["ci95"], v.mean(), v.std(ddof=1), ""])
for k, b in R["comparisons"].items():
    rows.append(["comparison", "", k, b["estimate"], *b["ci95"], "", "",
                 b["p_bootstrap_two_sided"]])

out = W / "stats_table_blast.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["quantity", "model", "condition", "estimate_seed42", "boot_ci95_low",
                "boot_ci95_high", "seed_mean", "seed_sd", "p_bootstrap"])
    for r in rows:
        w.writerow(r[:3] + [f"{x:.4f}" if isinstance(x, float) else x for x in r[3:]])
print("saved", out)

ident = {}
D = np.load(W / "preds_D.npz")
for s in SPLITS:
    i = D[f"ident__D__{s}"]
    v = i[~np.isnan(i)]
    ident[s] = {"coverage": float((~np.isnan(i)).mean()),
                "max_identity": float(v.max()), "p95_identity": float(np.percentile(v, 95)),
                "median_identity": float(np.median(v)),
                "share_ge40": float((i >= 40).mean()), "share_ge50": float((i >= 50).mean())}

(W / "blast_results.json").write_text(json.dumps({
    "method": {"tool": "DIAMOND 2.1.10 blastp --very-sensitive",
               "evalue": 1e-3, "rule": "best bit score among training sequences; no hit = no prediction (wrong)",
               "evalue_sensitivity": "e <= 10 gives strict-split macro-F1 0.838 vs 0.827 (coverage 0.881 vs 0.785)"},
    "bootstrap": R, "complementarity_and_identity_bins": AN,
    "nearest_training_identity_by_split": ident,
    "seeds": {"A_B_C": S1, "C_D_hybrid": S2},
}, indent=1))
print("saved blast_results.json")
