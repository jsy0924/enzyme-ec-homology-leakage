"""
효소 기능 분류 본 실험
=====================

연구 질문:
  훈련 데이터와 닮은 서열이 없을 때에도 단백질 언어모델 임베딩이
  유용한가? 상동성 누출을 제거하면 성능이 얼마나 떨어지는가?

모델:
  A  아미노산 조성 (20차원)        - 기준선
  B  ESM-2 임베딩 + 로지스틱 회귀   - 기본값 8M, --esm 으로 변경

분할:
  random        무작위 5-fold            (상동성 누출 있음)
  cluster@T     k-mer 클러스터 5-fold     (누출 제거, T = 유사도 기준치)

실행:
  python run_main.py                  전체 실행
  python run_main.py --n 300          클래스당 300개로 빠르게
  python run_main.py --esm 650M       ESM-2 650M 사용 (느림)
  python run_main.py --refetch        서열 다시 내려받기
  python run_main.py --reembed        임베딩 다시 계산

중간 결과는 캐시됩니다. 중단 후 다시 실행하면 이어서 진행합니다.
"""

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).parent
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

EC_NAMES = {
    1: "Oxidoreductase", 2: "Transferase", 3: "Hydrolase", 4: "Lyase",
    5: "Isomerase", 6: "Ligase", 7: "Translocase",
}
EC_CLASSES = list(EC_NAMES.keys())

ESM_MODELS = {
    "8M":   "facebook/esm2_t6_8M_UR50D",
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}

MIN_LEN, MAX_LEN = 100, 1000
RANDOM_SEED = 42
N_FOLDS = 5
KMER_K = 4
THRESHOLDS = [0.30, 0.20, 0.10, 0.05]   # 느슨 -> 엄격 (낮을수록 더 많이 묶임)
HTTP_RETRIES = 4                         # 끊긴 연결 재시도 횟수

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

LOG_LINES = []
META = {}


def log(msg=""):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)
    LOG_LINES.append(str(msg))


def safe_write(path, writer, quiet=False):
    """파일이 Excel 등에 잠겨 있어도 실행이 죽지 않게 한다."""
    path = Path(path)
    for attempt in range(4):
        target = path if attempt == 0 else path.with_name(
            f"{path.stem}_{attempt}{path.suffix}")
        try:
            writer(target)
            if attempt and not quiet:
                log(f"  [주의] {path.name} 이(가) 열려 있어 {target.name} 으로 저장")
            return target
        except (PermissionError, OSError):
            continue
    if not quiet:
        log(f"  [경고] {path.name} 저장 실패 (파일 잠김). 계속 진행합니다.")
    return None


def save_log():
    safe_write(OUT_DIR / "main_log.txt",
               lambda p: p.write_text("\n".join(LOG_LINES), encoding="utf-8"),
               quiet=True)


# ===================================================== 1. 데이터
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,ec,sequence,protein_name,organism_name"


def _clean(df, ec_digit):
    import pandas as pd
    df = df.rename(columns={
        "Entry": "accession", "EC number": "ec", "Sequence": "sequence",
        "Protein names": "name", "Organism": "organism",
    })
    if "ec" not in df or "sequence" not in df:
        return pd.DataFrame()
    df = df.dropna(subset=["ec", "sequence"])
    df = df[~df["ec"].astype(str).str.contains(";")]           # 단일 EC만
    df = df[df["ec"].astype(str).str.strip().str.startswith(f"{ec_digit}.")]
    df = df[~df["sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)]
    return df


def _get_with_retry(url, params):
    """연결이 끊겨도 몇 번 다시 시도한다. UniProt 수집 중 끊김이 잦다."""
    import requests

    last = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            wait = 3 * (attempt + 1)
            if attempt < HTTP_RETRIES - 1:
                print(f"\r    연결 실패 ({type(e).__name__}), "
                      f"{wait}초 후 재시도 {attempt+2}/{HTTP_RETRIES}   ",
                      end="", flush=True)
                time.sleep(wait)
    raise last


def fetch_ec_class(ec_digit, n):
    """EC 1단계가 ec_digit인 Swiss-Prot 단백질을 n개까지. 페이지 넘김 지원."""
    import pandas as pd

    query = (f"(ec:{ec_digit}.*) AND (reviewed:true) "
             f"AND (length:[{MIN_LEN} TO {MAX_LEN}]) AND (fragment:false)")
    url = UNIPROT
    params = {"query": query, "format": "tsv", "fields": FIELDS, "size": 500}
    frames, kept = [], 0

    while url and kept < n:
        r = _get_with_retry(url, params)
        if not META.get("uniprot_release"):
            META["uniprot_release"] = r.headers.get("x-uniprot-release", "unknown")
            META["uniprot_release_date"] = r.headers.get(
                "x-uniprot-release-date", "unknown")
        chunk = pd.read_csv(io.StringIO(r.text), sep="\t")
        if chunk.empty:
            break
        cleaned = _clean(chunk, ec_digit)
        frames.append(cleaned)
        kept += len(cleaned)
        nxt = r.links.get("next", {}).get("url")
        url, params = nxt, None          # 다음 링크에는 params를 다시 붙이지 않는다
        if url:
            time.sleep(0.3)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["sequence"])
    df["label"] = ec_digit
    return df.head(n).reset_index(drop=True)


def collect_data(n_per_class, refetch):
    """클래스별로 따로 캐시한다. 중간에 끊겨도 다음 실행에서 이어받는다."""
    import pandas as pd

    cache_dir = OUT_DIR / "seq_cache"
    cache_dir.mkdir(exist_ok=True)

    # 릴리스 정보는 한 번 받으면 캐시에 남겨 둔다 (전부 캐시로 읽는 날을 위해)
    meta_file = cache_dir / "uniprot_meta.json"
    if meta_file.exists():
        try:
            META.update(json.loads(meta_file.read_text(encoding="utf-8")))
        except Exception:
            pass

    log(f"[1/6] UniProt에서 서열 수집 (클래스당 최대 {n_per_class}개)")
    frames, failed = [], []

    for ec in EC_CLASSES:
        cf = cache_dir / f"ec{ec}_{n_per_class}.csv"
        if cf.exists() and not refetch:
            sub = pd.read_csv(cf)
            log(f"  EC {ec} ({EC_NAMES[ec]:>14}): {len(sub):5d}개  (캐시)")
            frames.append(sub)
            continue

        t0 = time.time()
        try:
            sub = fetch_ec_class(ec, n_per_class)
        except Exception as e:
            print("\r" + " " * 70 + "\r", end="")
            log(f"  EC {ec} ({EC_NAMES[ec]:>14}): 실패 - {type(e).__name__}")
            failed.append(ec)
            continue
        print("\r" + " " * 70 + "\r", end="")
        log(f"  EC {ec} ({EC_NAMES[ec]:>14}): {len(sub):5d}개  "
            f"({time.time()-t0:.0f}초)")
        if len(sub):
            safe_write(cf, lambda p: sub.to_csv(p, index=False,
                                                encoding="utf-8"), quiet=True)
            frames.append(sub)
        else:
            failed.append(ec)

    if failed:
        log("")
        log(f"  [경고] 받지 못한 클래스: {failed}")
        log("         인터넷이 끊겼을 가능성이 큽니다. 받은 클래스는 캐시에")
        log("         저장됐으니, 이 파일을 다시 실행하면 나머지만 받습니다.")
        if len(frames) < 2:
            log("\n[중단] 클래스가 2개 미만이라 분류를 할 수 없습니다.")
            save_log()
            sys.exit(1)

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    lens = data["sequence"].str.len()
    log("")
    log(f"  총 {len(data)}개 서열")
    log(f"  길이: 최소 {lens.min()}, 중앙값 {int(lens.median())}, 최대 {lens.max()}")
    if META.get("uniprot_release"):
        safe_write(meta_file, lambda p: p.write_text(
            json.dumps(META, ensure_ascii=False), encoding="utf-8"), quiet=True)
    log(f"  UniProt 릴리스: {META.get('uniprot_release','unknown')} "
        f"({META.get('uniprot_release_date','unknown')})")
    log("")
    counts = data["label"].value_counts().sort_index()
    for ec, c in counts.items():
        bar = "#" * max(1, int(40 * c / counts.max()))
        log(f"    EC {ec} {EC_NAMES[ec]:>14}: {bar} ({c})")
    if counts.min() < counts.max() * 0.5:
        log("")
        log("  [참고] 클래스 간 수량 차이가 큽니다. macro-F1 과 class_weight")
        log("         ='balanced' 로 보정하지만, 소수 클래스 성능은 낮게 나옵니다.")

    safe_write(OUT_DIR / "sequences_main.csv",
               lambda p: data.to_csv(p, index=False, encoding="utf-8"))
    log("  저장: sequences_main.csv")
    log("")
    return data


# ===================================================== 2. 특징
def composition_features(seq):
    import numpy as np
    counts = np.zeros(20)
    n = 0
    for ch in seq:
        i = AA_INDEX.get(ch)
        if i is not None:
            counts[i] += 1
            n += 1
    return counts / n if n else counts


def build_model_a(data):
    import numpy as np
    log("[2/6] 모델 A - 아미노산 조성")
    X = np.vstack([composition_features(s) for s in data["sequence"]])
    assert np.allclose(X.sum(axis=1), 1.0)
    log(f"  특징 행렬: {X.shape}")
    log("")
    return X


def build_model_b(data, esm_key, reembed):
    import numpy as np

    model_name = ESM_MODELS[esm_key]
    cache = OUT_DIR / f"emb_{esm_key}_{len(data)}.npz"
    if cache.exists() and not reembed:
        X = np.load(cache)["X"]
        if X.shape[0] == len(data):
            log(f"[3/6] 캐시 사용: {cache.name}  {X.shape}")
            log("      다시 계산하려면 --reembed")
            log("")
            return X

    import torch
    from transformers import AutoModel, AutoTokenizer

    log(f"[3/6] 모델 B - ESM-2 {esm_key} 임베딩")
    log(f"  {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm = AutoModel.from_pretrained(model_name).to(device).eval()
    log(f"  {sum(p.numel() for p in esm.parameters())/1e6:.1f}M 파라미터, {device}")
    log("  (ESM-2는 학습하지 않고 특징 추출기로만 사용)")

    seqs = data["sequence"].tolist()
    # 길이가 비슷한 것끼리 묶으면 패딩이 줄어 빨라진다
    order = np.argsort([len(s) for s in seqs])
    bs = 8 if esm_key in ("650M", "150M") else 16
    out = np.zeros((len(seqs), esm.config.hidden_size), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            batch = [seqs[j] for j in idx]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LEN + 2)
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = esm(**enc).last_hidden_state
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0                                   # cls 제외
            lengths = enc["attention_mask"].sum(dim=1)
            mask[torch.arange(mask.size(0), device=device), lengths - 1] = 0
            m = mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            out[idx] = pooled.cpu().numpy()

            done = min(i + bs, len(order))
            el = time.time() - t0
            eta = el / done * (len(order) - done)
            print(f"\r  임베딩 {done}/{len(order)}  경과 {el/60:.1f}분  "
                  f"남음 약 {eta/60:.1f}분   ", end="", flush=True)
    print()
    log(f"  완료: {out.shape}  ({(time.time()-t0)/60:.1f}분)")
    safe_write(cache, lambda p: np.savez_compressed(p, X=out))
    log(f"  저장: {cache.name}")
    log("")
    return out


# ===================================================== 3. 클러스터링
def kmer_matrix(seqs, k=KMER_K):
    import numpy as np
    from scipy.sparse import csr_matrix
    vocab, indptr, indices = {}, [0], []
    for s in seqs:
        for km in {s[i:i + k] for i in range(len(s) - k + 1)}:
            indices.append(vocab.setdefault(km, len(vocab)))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return csr_matrix((data, indices, indptr), shape=(len(seqs), len(vocab)))


def cluster_by_components(M, threshold, block=500):
    """유사도가 기준치 이상인 쌍을 모두 이어 연결 요소로 묶는다.

    대표 서열에만 견주는 탐욕적 방식과 달리, 기준치를 넘는 모든 쌍이
    반드시 같은 덩어리에 들어간다. 따라서 덩어리 단위로 나누면
    "훈련-테스트 사이에 기준치 이상인 쌍이 없다"가 수학적으로 보장된다.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = M.shape[0]
    sizes = np.asarray(M.sum(axis=1)).ravel()
    rows, cols = [], []
    for s in range(0, n, block):
        e = min(s + block, n)
        inter = (M[s:e] @ M.T).toarray()
        union = sizes[s:e, None] + sizes[None, :] - inter
        J = inter / np.maximum(union, 1)
        for r in range(e - s):             # 위쪽 삼각만 (자기 자신·중복 제외)
            J[r, : s + r + 1] = 0
        ii, jj = np.nonzero(J >= threshold)
        rows.extend((ii + s).tolist())
        cols.extend(jj.tolist())

    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels, len(rows)


def max_cross_similarity(M, idx_a, idx_b, block=500):
    """A와 B 사이 최대 Jaccard 유사도 (누출 검사용)."""
    import numpy as np
    sizes = np.asarray(M.sum(axis=1)).ravel()
    A, B = M[idx_a], M[idx_b]
    sa, sb = sizes[idx_a], sizes[idx_b]
    best = 0.0
    for s in range(0, A.shape[0], block):
        e = min(s + block, A.shape[0])
        inter = (A[s:e] @ B.T).toarray()
        union = sa[s:e, None] + sb[None, :] - inter
        best = max(best, float((inter / np.maximum(union, 1)).max()))
    return best


def build_splits(data, y):
    """무작위 분할 + 기준치별 클러스터 분할을 만든다."""
    import numpy as np
    from collections import Counter
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    seqs = data["sequence"].tolist()
    log(f"[4/6] 분할 구성 (k-mer k={KMER_K})")
    t0 = time.time()
    M = kmer_matrix(seqs)
    log(f"  k-mer 행렬 {M.shape}  ({time.time()-t0:.1f}초)")

    splits = {}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits["random"] = {
        "folds": list(skf.split(np.zeros(len(y)), y)),
        "n_clusters": len(y), "threshold": None,
    }

    for t in THRESHOLDS:
        t1 = time.time()
        cl, n_edges = cluster_by_components(M, t)
        sizes = Counter(cl.tolist())
        sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True,
                                    random_state=RANDOM_SEED)
        folds = list(sgkf.split(np.zeros(len(y)), y, groups=cl))
        big = max(sizes.values())
        splits[f"cluster@{t}"] = {
            "folds": folds, "n_clusters": len(sizes), "threshold": t,
            "clusters": cl, "largest_cluster": int(big),
            "n_similar_pairs": int(n_edges),
        }
        log(f"  기준치 {t:.2f}: 클러스터 {len(sizes):5d}개  "
            f"유사쌍 {n_edges:5d}개  "
            f"최대덩어리 {big}개({big/len(y)*100:.1f}%)  "
            f"단독 {sum(1 for v in sizes.values() if v==1)}개  "
            f"[{time.time()-t1:.0f}초]")
        if big > 0.3 * len(y):
            log("    [경고] 한 덩어리가 전체의 30%를 넘습니다. 연쇄 결합이")
            log("           일어났을 수 있으니 기준치를 높이세요.")
    log("")

    # ---- 누출 검사 ----
    log("  [누출 검사] 각 분할에서 훈련-테스트 최대 유사도")
    all_clean = True
    for name, s in splits.items():
        worst = 0.0
        for tr, te in s["folds"]:
            worst = max(worst, max_cross_similarity(M, tr, te))
        s["max_train_test_jaccard"] = worst
        if s["threshold"] is None:
            verdict = "(누출 있음 - 의도된 비교군)"
        elif worst < s["threshold"]:
            verdict = "O 기준치 미만 - 깨끗"
        else:
            verdict = "X 기준치 이상 - 버그"
            all_clean = False
        log(f"    {name:<16} 최대 Jaccard = {worst:.3f}   {verdict}")
    if not all_clean:
        log("")
        log("  [중요] 클러스터 분할에 누출이 남아 있습니다. 결과를 믿지 마세요.")
    log("")
    return splits


# ===================================================== 4. 평가
def evaluate(features, y, splits):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make_clf():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
        )

    log("[5/6] 평가")
    results, preds = {}, {}

    for split_name, s in splits.items():
        for model_name, X in features.items():
            p = np.empty(len(y), dtype=y.dtype)
            for tr, te in s["folds"]:
                clf = make_clf()
                clf.fit(X[tr], y[tr])
                p[te] = clf.predict(X[te])
            key = (model_name, split_name)
            preds[key] = p
            results[key] = {
                "accuracy": float(accuracy_score(y, p)),
                "macro_F1": float(f1_score(y, p, average="macro",
                                           zero_division=0)),
            }

    # 셔플 대조군 (무작위 분할, 가장 강한 모델)
    strongest = list(features)[-1]
    y_shuf = np.random.RandomState(0).permutation(y)
    p = np.empty(len(y), dtype=y.dtype)
    for tr, te in splits["random"]["folds"]:
        clf = make_clf()
        clf.fit(features[strongest][tr], y_shuf[tr])
        p[te] = clf.predict(features[strongest][te])
    shuf_f1 = float(f1_score(y_shuf, p, average="macro", zero_division=0))

    # ---- 표 출력 ----
    chance = 1.0 / len(np.unique(y))
    model_names = list(features)
    log("")
    log("  macro-F1")
    header = f"  {'분할':<16}" + "".join(f"{m:>22}" for m in model_names)
    log(header)
    log("  " + "-" * (16 + 22 * len(model_names)))
    for split_name in splits:
        row = f"  {split_name:<16}"
        for m in model_names:
            row += f"{results[(m, split_name)]['macro_F1']:>22.3f}"
        log(row)
    log("")
    log(f"  라벨 셔플 대조군 = {shuf_f1:.3f}   무작위 추측 = {chance:.3f}")

    # ---- 자동 해석 ----
    log("")
    log("  [점검]")
    if shuf_f1 > chance + 0.10:
        log(f"  X 셔플 대조군이 {shuf_f1:.3f}로 높습니다. 코드 누수 의심.")
    else:
        log(f"  O 셔플 대조군 {shuf_f1:.3f} - 정상")

    strict = f"cluster@{min(THRESHOLDS)}"
    for m in model_names:
        r = results[(m, "random")]["macro_F1"]
        c = results[(m, strict)]["macro_F1"]
        log(f"  - {m}: 무작위 {r:.3f} -> 엄격 클러스터 {c:.3f}  ({c-r:+.3f})")
    if len(model_names) >= 2:
        a, b = model_names[0], model_names[-1]
        gap_r = (results[(b, "random")]["macro_F1"]
                 - results[(a, "random")]["macro_F1"])
        gap_c = (results[(b, strict)]["macro_F1"]
                 - results[(a, strict)]["macro_F1"])
        log(f"  - {b} 대 {a} 격차: 무작위 {gap_r:+.3f} -> 엄격 {gap_c:+.3f}")
        if gap_c > gap_r:
            log("    분할이 엄격해질수록 언어모델의 이점이 커집니다.")
        else:
            log("    분할이 엄격해질수록 언어모델의 이점이 줄어듭니다.")

    return results, preds, shuf_f1


# ===================================================== 5. 그림
def make_plots(features, y, splits, results, preds):
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except Exception as e:
        log(f"  그림 생성 건너뜀: {e}")
        return

    log("")
    log("[6/6] 그림")
    model_names = list(features)
    split_names = list(splits)
    xlabels = ["random\n(leakage)"] + [
        f"cluster\nJ>={s['threshold']}" for n, s in splits.items()
        if s["threshold"] is not None]

    # --- 핵심 그림: 분할 엄격도에 따른 성능 ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = ["#888888", "#1f4e9c", "#d2691e", "#2e8b57"]
    for i, m in enumerate(model_names):
        vals = [results[(m, s)]["macro_F1"] for s in split_names]
        ax.plot(range(len(split_names)), vals, "o-", linewidth=2.2,
                markersize=8, label=m, color=colors[i % len(colors)])
    ax.axhline(1.0 / len(np.unique(y)), ls="--", c="#bbb", lw=1.2)
    ax.text(len(split_names) - 1, 1.0 / len(np.unique(y)) + 0.008,
            "chance", ha="right", fontsize=9, color="#999")
    ax.set_xticks(range(len(split_names)))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("macro-F1")
    ax.set_xlabel("split: looser  ->  stricter (less homology leakage)")
    ax.set_title("Performance drops as homology leakage is removed")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    safe_write(OUT_DIR / "fig1_split_effect.png",
               lambda p: plt.savefig(p, dpi=160, bbox_inches="tight"))
    plt.close()
    log("  fig1_split_effect.png  (핵심 그림)")

    # --- 혼동 행렬: 가장 강한 모델, 무작위 vs 엄격 ---
    m = model_names[-1]
    strict = f"cluster@{min(THRESHOLDS)}"
    labels = sorted(np.unique(y))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sname, tag in zip(axes, ["random", strict],
                              ["random split (leakage)", "cluster split"]):
        cm = confusion_matrix(y, preds[(m, sname)], labels=labels,
                              normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"EC{l}" for l in labels])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([f"EC{l}" for l in labels])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{m} - {tag}\nmacro-F1 = "
                     f"{results[(m, sname)]['macro_F1']:.3f}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j] > 0.01:
                    ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if cm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    safe_write(OUT_DIR / "fig2_confusion.png",
               lambda p: plt.savefig(p, dpi=160, bbox_inches="tight"))
    plt.close()
    log("  fig2_confusion.png")


# ===================================================== main
def main():
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="클래스당 서열 수")
    ap.add_argument("--esm", default="8M",
                    help="쉼표로 여러 개 가능. 예: --esm 8M,650M")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--reembed", action="store_true")
    ap.add_argument("--only-a", action="store_true", help="torch 없이 모델 A만")
    args = ap.parse_args()

    esm_keys = [k.strip() for k in args.esm.split(",") if k.strip()]
    bad = [k for k in esm_keys if k not in ESM_MODELS]
    if bad and not args.only_a:
        print(f"알 수 없는 ESM 모델: {bad}. 가능한 값: {list(ESM_MODELS)}")
        sys.exit(1)

    np.random.seed(RANDOM_SEED)

    log("=" * 66)
    log("효소 기능 분류 본 실험")
    log(f"실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"설정: 클래스당 {args.n}개, ESM-2 {esm_keys}, "
        f"k-mer k={KMER_K}, 기준치 {THRESHOLDS}")
    log("=" * 66)
    log(f"python {sys.version.split()[0]}")
    if not args.only_a:
        try:
            import torch
            log("GPU: " + (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else "없음 (CPU)"))
        except ImportError:
            log("[중단] torch 가 없습니다. --only-a 로 모델 A만 돌리거나,")
            log("       2_install_torch.bat 을 실행하세요.")
            save_log()
            sys.exit(1)
    log("")

    data = collect_data(args.n, args.refetch)
    y = data["label"].to_numpy()

    features = {"A: composition": build_model_a(data)}
    if not args.only_a:
        for i, key in enumerate(esm_keys):
            label = f"{chr(ord('B') + i)}: ESM-2 {key}"
            features[label] = build_model_b(data, key, args.reembed)
    else:
        log("[3/6] 모델 B 건너뜀 (--only-a)")
        log("")

    splits = build_splits(data, y)
    results, preds, shuf_f1 = evaluate(features, y, splits)
    make_plots(features, y, splits, results, preds)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "n_per_class": args.n, "esm": esm_keys, "kmer_k": KMER_K,
            "thresholds": THRESHOLDS, "n_folds": N_FOLDS,
            "min_len": MIN_LEN, "max_len": MAX_LEN, "seed": RANDOM_SEED,
            "n_sequences": int(len(data)),
        },
        "uniprot": {
            "release": META.get("uniprot_release", "unknown"),
            "release_date": META.get("uniprot_release_date", "unknown"),
        },
        "classes_present": sorted(int(c) for c in set(y.tolist())),
        "splits": {
            name: {"n_clusters": int(s["n_clusters"]),
                   "threshold": s["threshold"],
                   "largest_cluster": s.get("largest_cluster"),
                   "n_similar_pairs": s.get("n_similar_pairs"),
                   "max_train_test_jaccard": s.get("max_train_test_jaccard")}
            for name, s in splits.items()
        },
        "results": {f"{m} | {s}": v for (m, s), v in results.items()},
        "shuffle_control": shuf_f1,
    }
    safe_write(OUT_DIR / "results_main.json",
               lambda p: p.write_text(
                   json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8"))

    log("")
    log("=" * 66)
    log("완료. 생성된 파일:")
    log("  sequences_main.csv      수집한 서열 (캐시)")
    for key in ([] if args.only_a else esm_keys):
        log(f"  emb_{key}_{len(data)}.npz       ESM-2 {key} 임베딩 (캐시)")
    log("  results_main.json       모든 수치")
    log("  fig1_split_effect.png   핵심 그림")
    log("  fig2_confusion.png      혼동 행렬")
    log("  main_log.txt            이 로그 전체")
    log("=" * 66)
    save_log()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n[중단됨] 캐시는 남아 있으니 다시 실행하면 이어서 진행합니다.")
        save_log()
    except Exception:
        import traceback
        log("\n[오류 발생]")
        log(traceback.format_exc())
        save_log()
        print("\nmain_log.txt 를 Claude에게 보여주세요.")
        sys.exit(1)
