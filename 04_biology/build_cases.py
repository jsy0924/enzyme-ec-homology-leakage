"""Case lists for the biological follow-up.

Idea 1 (strict split, cluster@0.05): where BLAST and ESM-2 650M disagree.
Idea 2 (all 7,000 x 7,000 DIAMOND pairs): sequence identity vs function.

Sampling rule (fixed before looking at any case):
  idea 1: for each of 4 disagreement categories and each EC class, draw up to
          2 sequences at random (seed 2026). The first draw is round 1, the
          second is round 2.
  idea 2: all 37 high-identity / different-class pairs; 20 of the 62 partial
          matches drawn at random (seed 2026).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/blast")
import blast_predict as BP  # noqa: E402

SRC = Path("/mnt/user-data/uploads/Documents/enzyme-project")
OUT = Path("/home/claude/bio/out")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026
STRICT = "cluster@0.05"
EC_NAME = {1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase", 4: "Lyase",
           5: "Isomerase", 6: "Ligase", 7: "Translocase"}

df = pd.read_csv(SRC / "sequences_main.csv")
df["ec"] = df["ec"].astype(str).str.strip()
y = df["label"].to_numpy()
acc = df["accession"].to_numpy()
P = np.load("/home/claude/analysis/preds.npz")
fold = P[f"fold__{STRICT}"]
pred_B = P[f"pred__B__{STRICT}"]
pred_C = P[f"pred__C__{STRICT}"]


def uniprot(a):
    return f"https://www.uniprot.org/uniprotkb/{a}/entry"


# ---------- nearest TRAINING hit for every sequence under the strict split ----------
near_idx = np.full(len(y), -1)
near_id = np.full(len(y), np.nan)
pred_D = np.zeros(len(y), dtype=int)
for k in np.unique(fold):
    in_train = fold != k
    for i in np.where(fold == k)[0]:
        si, pid, _ = BP.TARGETS.get(i, (np.empty(0, int), np.empty(0), np.empty(0)))
        ok = in_train[si] if len(si) else np.empty(0, bool)
        if ok.any():
            j = int(np.argmax(ok))
            near_idx[i], near_id[i], pred_D[i] = si[j], pid[j], y[si[j]]
saved = np.load("/home/claude/blast/preds_D.npz")[f"pred__D__{STRICT}"]
assert (pred_D == saved).all(), "BLAST predictions do not match the saved run"

# orphan = no DIAMOND hit to ANY other sequence in the 7,000
hits = pd.read_csv("/home/claude/blast/hits.tsv", sep="\t", header=None,
                   names=["q", "s", "pid", "len", "ev", "bit", "qcov", "scov"])
hits = hits[hits.q != hits.s]
has_hit = set(hits.q) | set(hits.s)
orphan = ~df["accession"].isin(has_hit).to_numpy()

# ---------- idea 1 ----------
cat = np.full(len(y), "", dtype=object)
cat[(pred_D == 0) & (pred_C == y)] = "A_BLAST no hit, 650M right"
cat[(pred_D == 0) & (pred_C != y)] = "B_BLAST no hit, 650M wrong"
cat[(pred_D > 0) & (pred_D == y) & (pred_C != y)] = "C_BLAST right, 650M wrong"
cat[(pred_D > 0) & (pred_D != y) & (pred_C == y)] = "D_BLAST wrong, 650M right"
cat[(pred_D > 0) & (pred_D != y) & (pred_C != y)] = "E_both answered, both wrong"
cat[(pred_D == y) & (pred_C == y)] = "F_both right"

i1 = pd.DataFrame({
    "category": cat,
    "accession": acc,
    "uniprot_url": [uniprot(a) for a in acc],
    "protein_name": df["name"],
    "organism": df["organism"],
    "EC_full": df["ec"],
    "true_class": y,
    "pred_650M": pred_C,
    "pred_8M": pred_B,
    "pred_BLAST": pred_D,
    "BLAST_nearest_train_accession": [acc[j] if j >= 0 else "" for j in near_idx],
    "BLAST_nearest_train_name": [df["name"].iat[j] if j >= 0 else "" for j in near_idx],
    "BLAST_nearest_train_EC": [df["ec"].iat[j] if j >= 0 else "" for j in near_idx],
    "BLAST_nearest_identity_pct": np.round(near_id, 1),
    "orphan_in_dataset": np.where(orphan, "yes", "no"),
    "length": df["sequence"].str.len(),
})
i1.to_csv(OUT / "idea1_all_7000_strict_split.csv", index=False, encoding="utf-8-sig")

rng = np.random.default_rng(SEED)
picks = []
for c in ["A_BLAST no hit, 650M right", "B_BLAST no hit, 650M wrong",
          "C_BLAST right, 650M wrong", "D_BLAST wrong, 650M right"]:
    for ec in range(1, 8):
        pool = np.where((cat == c) & (y == ec))[0]
        chosen = rng.permutation(pool)[:2]
        for r, i in enumerate(chosen, start=1):
            picks.append((c, ec, r, i))
sample = i1.iloc[[p[3] for p in picks]].copy()
sample.insert(0, "round", [p[2] for p in picks])
sample.insert(0, "case_id", [f"1-{n+1:02d}" for n in range(len(picks))])
sample.to_csv(OUT / "idea1_sample.csv", index=False, encoding="utf-8-sig")

# ---------- idea 2 ----------
h = hits[hits.q < hits.s].copy()
info = df.set_index("accession")
for side in ("q", "s"):
    h[f"class_{side}"] = h[side].map(info["label"])
    h[f"EC_{side}"] = h[side].map(info["ec"])
    h[f"name_{side}"] = h[side].map(info["name"])
    h[f"org_{side}"] = h[side].map(info["organism"])


def pairs_table(t):
    return pd.DataFrame({
        "accession_1": t.q, "uniprot_1": [uniprot(a) for a in t.q],
        "name_1": t.name_q, "organism_1": t.org_q, "EC_1": t.EC_q, "class_1": t.class_q,
        "accession_2": t.s, "uniprot_2": [uniprot(a) for a in t.s],
        "name_2": t.name_s, "organism_2": t.org_s, "EC_2": t.EC_s, "class_2": t.class_s,
        "identity_pct": t.pid, "aligned_len": t.len,
        "coverage_1_pct": t.qcov, "coverage_2_pct": t.scov, "evalue": t.ev,
    }).reset_index(drop=True)


full = h[(h.qcov >= 80) & (h.scov >= 80)]
hi_diff = pairs_table(full[(full.pid >= 50) & (full.class_q != full.class_s)]
                      .sort_values("pid", ascending=False))
hi_diff.insert(0, "pair_id", [f"2H-{n+1:02d}" for n in range(len(hi_diff))])
hi_diff.to_csv(OUT / "idea2_high_identity_different_class.csv", index=False, encoding="utf-8-sig")

part = h[(h.pid >= 50) & ((h.qcov < 50) | (h.scov < 50)) & (h.class_q != h.class_s)]
part_t = pairs_table(part.sort_values("pid", ascending=False))
part_t.to_csv(OUT / "idea2_partial_match_different_class_all.csv", index=False, encoding="utf-8-sig")
part_s = part_t.iloc[np.sort(np.random.default_rng(SEED).permutation(len(part_t))[:20])].copy()
part_s.insert(0, "pair_id", [f"2P-{n+1:02d}" for n in range(len(part_s))])
part_s.to_csv(OUT / "idea2_partial_match_sample.csv", index=False, encoding="utf-8-sig")


# ---------- identity vs EC level (full-length pairs) ----------
def ec_levels_shared(a, b):
    pa, pb = a.split("."), b.split(".")
    n = 0
    for x, z in zip(pa, pb):
        if x == z and x not in ("-", "n"):
            n += 1
        else:
            break
    return n


full = full.copy()
full["shared"] = [ec_levels_shared(a, b) for a, b in zip(full.EC_q, full.EC_s)]
bins = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
rows = []
for lo, hi in bins:
    t = full[(full.pid >= lo) & (full.pid < hi)]
    rows.append({"identity_bin": f"{lo}-{min(hi,100)}%", "n_pairs": len(t),
                 **{f"same_EC_level{L}": round((t.shared >= L).mean(), 3) for L in range(1, 5)}})
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "idea2_identity_vs_EC_level.csv", index=False, encoding="utf-8-sig")

print("idea 1 category counts:")
print(pd.Series(cat).value_counts().sort_index().to_string())
print(f"\nidea 1 sample: {len(sample)} cases (round 1: {(sample['round']==1).sum()})")
print(sample.groupby('category').size().to_string())
print(f"orphans among sample: {(sample.orphan_in_dataset=='yes').sum()}")
print(f"\nidea 2: high-identity/different-class pairs {len(hi_diff)}, "
      f"partial-match/different-class {len(part_t)} (sample {len(part_s)})")
print("\nidentity vs shared EC levels (full-length pairs):")
print(summ.to_string(index=False))
