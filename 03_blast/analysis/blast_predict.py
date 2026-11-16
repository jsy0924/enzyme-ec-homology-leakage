"""Model D: BLAST-style nearest-neighbour label transfer (DIAMOND blastp).

For every test sequence, take the highest-bit-score hit among that fold's
TRAINING sequences and copy its EC class. If no training sequence passes
e-value <= 1e-3, there is no prediction (label 0), which counts as wrong.

Search is all-vs-all once; restricting to training targets per fold is done
here. Ranking uses bit score, which does not depend on database size.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/mnt/user-data/uploads/Documents/enzyme-project")
WORK = Path("/home/claude/blast")
ANALYSIS = Path("/home/claude/analysis")

df = pd.read_csv(SRC / "sequences_main.csv")
ACC = df["accession"].tolist()
IDX = {a: i for i, a in enumerate(ACC)}
N = len(ACC)

hits = pd.read_csv(WORK / "hits.tsv", sep="\t", header=None,
                   names=["q", "s", "pident", "length", "evalue", "bitscore",
                          "qcov", "scov"])
hits = hits[hits.q != hits.s]                         # drop self-hits
hits["qi"] = hits.q.map(IDX)
hits["si"] = hits.s.map(IDX)
hits = hits.sort_values(["qi", "bitscore", "si"], ascending=[True, False, True])

TARGETS = defaultdict(lambda: (np.empty(0, int), np.empty(0), np.empty(0)))
for qi, g in hits.groupby("qi", sort=False):
    TARGETS[qi] = (g.si.to_numpy(), g.pident.to_numpy(), g.bitscore.to_numpy())


def predict(y, fold_of):
    """Return predicted label (0 = no hit) and best-hit identity (nan = no hit)."""
    pred = np.zeros(N, dtype=np.int64)
    ident = np.full(N, np.nan)
    for k in np.unique(fold_of):
        in_train = fold_of != k
        for i in np.where(fold_of == k)[0]:
            si, pid, _ = TARGETS[i]
            ok = in_train[si] if len(si) else np.empty(0, bool)
            if ok.any():
                j = int(np.argmax(ok))                 # first = best bit score
                pred[i] = y[si[j]]
                ident[i] = pid[j]
    return pred, ident


def main():
    P = np.load(ANALYSIS / "preds.npz")
    y = P["y"]
    assert len(y) == N
    out = {}
    for s in P["split_names"]:
        s = str(s)
        pred, ident = predict(y, P[f"fold__{s}"])
        out[f"pred__D__{s}"] = pred
        out[f"ident__D__{s}"] = ident
        cov = (pred > 0).mean()
        acc_hit = (pred[pred > 0] == y[pred > 0]).mean()
        from sklearn.metrics import f1_score
        f1 = f1_score(y, pred, labels=list(range(1, 8)), average="macro")
        print(f"D | {s:<13} macro-F1 {f1:.3f}   coverage {cov:.3f}   "
              f"accuracy when it answers {acc_hit:.3f}   "
              f"median best-hit identity {np.nanmedian(ident):.1f}%")
    np.savez_compressed(WORK / "preds_D.npz", **out)
    print("saved preds_D.npz")


if __name__ == "__main__":
    main()
