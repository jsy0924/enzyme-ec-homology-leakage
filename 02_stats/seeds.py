"""Seed repetition: redo the whole 5-fold CV with different fold-assignment seeds.

Clusters do not depend on the seed (they come from sequences only), so they are
built once. Only the assignment of sequences/clusters to the 5 folds changes.
This measures how much each number moves just because of how the data were cut.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/mnt/user-data/uploads/Documents/enzyme-project")
WORK = Path("/home/claude/analysis")
sys.path.insert(0, "/home/claude")
import run_main as R  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def make_clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))


def main():
    data = pd.read_csv(SRC / "sequences_main.csv")
    y = data["label"].to_numpy()
    R.log = lambda *a, **k: None
    feats = {
        "A": R.build_model_a(data),
        "B": np.load(SRC / "emb_8M_7000.npz")["X"],
        "C": np.load(SRC / "emb_650M_7000.npz")["X"],
    }
    M = R.kmer_matrix(data["sequence"].tolist())
    clusters = {t: R.cluster_by_components(M, t)[0] for t in R.THRESHOLDS}
    split_names = ["random"] + [f"cluster@{t}" for t in R.THRESHOLDS]

    out = {}
    t_all = time.time()
    for seed in SEEDS:
        t0 = time.time()
        folds = {"random": list(StratifiedKFold(5, shuffle=True, random_state=seed)
                                .split(np.zeros(len(y)), y))}
        for t in R.THRESHOLDS:
            folds[f"cluster@{t}"] = list(
                StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                .split(np.zeros(len(y)), y, groups=clusters[t]))
        res = {}
        for s in split_names:
            for m, X in feats.items():
                p = np.empty(len(y), dtype=y.dtype)
                for tr, te in folds[s]:
                    p[te] = make_clf().fit(X[tr], y[tr]).predict(X[te])
                res[f"{m}|{s}"] = float(f1_score(y, p, average="macro"))
        out[str(seed)] = res
        (WORK / "seeds_results.json").write_text(json.dumps(out, indent=1))
        print(f"seed {seed:>2}: done in {time.time()-t0:.0f}s   "
              f"B drop {res['B|random']-res['B|cluster@0.05']:+.3f}   "
              f"C drop {res['C|random']-res['C|cluster@0.05']:+.3f}",
              flush=True)
    print(f"all seeds done in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
