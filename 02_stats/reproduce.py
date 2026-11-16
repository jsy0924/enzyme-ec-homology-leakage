"""Rebuild the exact splits and out-of-fold predictions from cached embeddings.

Uses run_main.py's own functions so the splits are identical to the logged run.
Saves everything the statistics and figures need into preds.npz.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/mnt/user-data/uploads/Documents/enzyme-project")
WORK = Path("/home/claude/analysis")
sys.path.insert(0, "/home/claude")

import run_main as R  # noqa: E402

R.OUT_DIR = WORK                     # never write into the user's folder
R.log = lambda *a, **k: None         # keep the console quiet

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def make_clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))


def main():
    data = pd.read_csv(SRC / "sequences_main.csv")
    y = data["label"].to_numpy()
    feats = {
        "A": R.build_model_a(data),
        "B": np.load(SRC / "emb_8M_7000.npz")["X"],
        "C": np.load(SRC / "emb_650M_7000.npz")["X"],
    }

    t0 = time.time()
    splits = R.build_splits(data, y)
    print(f"splits rebuilt in {time.time()-t0:.0f}s")
    for name, s in splits.items():
        print(f"  {name:<14} clusters={s['n_clusters']:5d}  "
              f"maxJ={s['max_train_test_jaccard']:.3f}")

    split_names = list(splits)
    preds = {}
    fold_of = {}
    for sname in split_names:
        f_id = np.empty(len(y), dtype=np.int64)
        for k, (tr, te) in enumerate(splits[sname]["folds"]):
            f_id[te] = k
        fold_of[sname] = f_id
        for m, X in feats.items():
            t1 = time.time()
            p = np.empty(len(y), dtype=y.dtype)
            for tr, te in splits[sname]["folds"]:
                clf = make_clf().fit(X[tr], y[tr])
                p[te] = clf.predict(X[te])
            preds[(m, sname)] = p
            f1 = f1_score(y, p, average="macro")
            print(f"  {m} | {sname:<14} macro-F1={f1:.3f}  ({time.time()-t1:.0f}s)",
                  flush=True)

    np.savez_compressed(
        WORK / "preds.npz",
        y=y,
        split_names=np.array(split_names),
        models=np.array(list(feats)),
        clusters_005=splits["cluster@0.05"]["clusters"],
        **{f"pred__{m}__{s}": preds[(m, s)] for (m, s) in preds},
        **{f"fold__{s}": fold_of[s] for s in split_names},
    )
    print("saved preds.npz")


if __name__ == "__main__":
    main()
