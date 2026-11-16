"""Seed repetition for BLAST (D) and the hybrid (BLAST hit, else ESM-2 650M).

Same 10 fold-assignment seeds as the earlier seed run. C is refit per seed
because the hybrid needs C's predictions on exactly the same folds.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude")
sys.path.insert(0, "/home/claude/blast")
import run_main as R  # noqa: E402
import blast_predict as BP  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SRC = Path("/mnt/user-data/uploads/Documents/enzyme-project")
WORK = Path("/home/claude/blast")
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
LAB = list(range(1, 8))


def main():
    R.log = lambda *a, **k: None
    data = pd.read_csv(SRC / "sequences_main.csv")
    y = data["label"].to_numpy()
    XC = np.load(SRC / "emb_650M_7000.npz")["X"]
    M = R.kmer_matrix(data["sequence"].tolist())
    clusters = {t: R.cluster_by_components(M, t)[0] for t in R.THRESHOLDS}
    splits = ["random"] + [f"cluster@{t}" for t in R.THRESHOLDS]

    out = {}
    for seed in SEEDS:
        t0 = time.time()
        res = {}
        for s in splits:
            if s == "random":
                folds = StratifiedKFold(5, shuffle=True, random_state=seed).split(
                    np.zeros(len(y)), y)
            else:
                t = float(s.split("@")[1])
                folds = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
                    np.zeros(len(y)), y, groups=clusters[t])
            fold_of = np.empty(len(y), dtype=np.int64)
            pc = np.empty(len(y), dtype=y.dtype)
            for k, (tr, te) in enumerate(folds):
                fold_of[te] = k
                clf = make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=3000, C=1.0, class_weight="balanced")).fit(XC[tr], y[tr])
                pc[te] = clf.predict(XC[te])
            pd_, _ = BP.predict(y, fold_of)
            hyb = np.where(pd_ > 0, pd_, pc)
            res[f"C|{s}"] = float(f1_score(y, pc, labels=LAB, average="macro"))
            res[f"D|{s}"] = float(f1_score(y, pd_, labels=LAB, average="macro"))
            res[f"D+C|{s}"] = float(f1_score(y, hyb, labels=LAB, average="macro"))
            res[f"Dcov|{s}"] = float((pd_ > 0).mean())
        out[str(seed)] = res
        (WORK / "seeds_D_results.json").write_text(json.dumps(out, indent=1))
        print(f"seed {seed:>2}: {time.time()-t0:.0f}s   "
              f"D {res['D|random']:.3f}->{res['D|cluster@0.05']:.3f}   "
              f"C {res['C|random']:.3f}->{res['C|cluster@0.05']:.3f}   "
              f"D+C {res['D+C|random']:.3f}->{res['D+C|cluster@0.05']:.3f}", flush=True)
    print("all seeds done")


if __name__ == "__main__":
    main()
