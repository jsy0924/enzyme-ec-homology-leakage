"""t-SNE coordinates for A, B, C with identical settings, plus a check that
does not depend on t-SNE at all: for the 556 orphan sequences (no DIAMOND hit
to any other sequence), what fraction of their nearest neighbours in the
ORIGINAL feature space share their EC class?

t-SNE settings (same for all three):
  standardize -> PCA to 50 dims (B, C; A has only 20 dims) -> t-SNE
  perplexity 30, init "pca", learning_rate "auto", random_state 42
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SRC = Path("/mnt/user-data/uploads/Documents/enzyme-project")
OUT = Path("/home/claude/tsne")
sys.path.insert(0, "/home/claude")
import run_main as R  # noqa: E402

R.log = lambda *a, **k: None
K_NN = 15


def main():
    df = pd.read_csv(SRC / "sequences_main.csv")
    y = df["label"].to_numpy()
    feats = {
        "A": R.build_model_a(df),
        "B": np.load(SRC / "emb_8M_7000.npz")["X"],
        "C": np.load(SRC / "emb_650M_7000.npz")["X"],
    }

    h = pd.read_csv("/home/claude/blast/hits.tsv", sep="\t", header=None,
                    usecols=[0, 1], names=["q", "s"])
    h = h[h.q != h.s]
    orphan = (~df["accession"].isin(set(h.q) | set(h.s))).to_numpy()

    coords, purity = {}, {}
    for m, X in feats.items():
        Z = StandardScaler().fit_transform(X)
        Zp = PCA(n_components=50, random_state=42).fit_transform(Z) if Z.shape[1] > 50 else Z

        # neighbour purity in the full standardized space (no t-SNE involved)
        nn = NearestNeighbors(n_neighbors=K_NN + 1).fit(Z)
        _, idx = nn.kneighbors(Z)
        same = (y[idx[:, 1:]] == y[:, None]).mean(axis=1)
        purity[m] = {"all": float(same.mean()),
                     "orphans": float(same[orphan].mean()),
                     "non_orphans": float(same[~orphan].mean())}

        t0 = time.time()
        coords[m] = TSNE(n_components=2, perplexity=30, init="pca",
                         learning_rate="auto", random_state=42).fit_transform(Zp)
        print(f"{m}: t-SNE {time.time()-t0:.0f}s   "
              f"{K_NN}-NN same-class share: all {purity[m]['all']:.3f}  "
              f"orphans {purity[m]['orphans']:.3f}", flush=True)

    np.savez_compressed(OUT / "tsne_coords.npz", y=y, orphan=orphan,
                        **{f"xy_{m}": coords[m] for m in coords})
    (OUT / "neighbour_purity.json").write_text(json.dumps(
        {"k": K_NN, "space": "standardized full feature space",
         "n_orphans": int(orphan.sum()), "purity": purity,
         "chance_by_class_share": 1 / 7}, indent=1))
    print("saved")


if __name__ == "__main__":
    main()
