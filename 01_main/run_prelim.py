"""
효소 기능 분류 예비실험 - 모델 A & B
====================================

모델 A: 아미노산 조성(20차원) + 로지스틱 회귀
모델 B: ESM-2 8M 임베딩(320차원) + 로지스틱 회귀
과제  : EC 1단계 7개 클래스 분류

실행 방법 (명령 프롬프트 또는 PowerShell):
    cd %USERPROFILE%\\Documents\\enzyme-project

    # 모델 A만 (torch 불필요, 설치 가벼움)
    python run_prelim.py --only-a

    # 모델 A + B (torch 필요)
    python run_prelim.py

처음 실행 전 한 번만 (중국에서는 미러 사용 권장):
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple requests scikit-learn matplotlib pandas
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch --index-url https://download.pytorch.org/whl/cpu
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers

주의: 이 단계의 숫자는 논문에 쓰지 마세요. 데이터가 적어 흔들리고,
      무작위 분할이라 상동성 누출이 그대로 있습니다.
"""

import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- 설정
N_PER_CLASS = 30      # 클래스당 서열 수. 처음에는 작게.
MIN_LEN     = 100     # 너무 짧은 단백질 제외
MAX_LEN     = 1000    # 너무 긴 단백질 제외 (메모리 절약)
RANDOM_SEED = 42
N_FOLDS     = 5

EC_NAMES = {
    1: "Oxidoreductase",
    2: "Transferase",
    3: "Hydrolase",
    4: "Lyase",
    5: "Isomerase",
    6: "Ligase",
    7: "Translocase",
}
EC_CLASSES = list(EC_NAMES.keys())

OUT_DIR = Path(__file__).parent
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Windows 콘솔 코드페이지가 한국어를 못 찍어도 죽지 않게 한다.
# 로그 파일(run_log.txt)은 항상 UTF-8로 온전히 저장된다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 실행 기록을 파일에도 남긴다 (Claude가 읽을 수 있도록)
LOG_LINES = []


def log(msg=""):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # 콘솔이 못 찍는 글자는 건너뛰고, 로그 파일에는 온전히 남긴다
        print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)
    LOG_LINES.append(str(msg))


def save_log():
    safe_write(OUT_DIR / "run_log.txt",
               lambda p: p.write_text("\n".join(LOG_LINES), encoding="utf-8"),
               quiet=True)


def safe_write(path, writer, quiet=False):
    """파일이 Excel 등에서 열려 있어 잠겨도 실행이 죽지 않게 한다.

    원래 이름이 막히면 _1, _2 ... 를 붙여 다른 이름으로 저장한다.
    writer(path) 는 실제 저장을 수행하는 함수.
    """
    path = Path(path)
    for attempt in range(4):
        target = path if attempt == 0 else path.with_name(
            f"{path.stem}_{attempt}{path.suffix}"
        )
        try:
            writer(target)
            if attempt and not quiet:
                log(f"  [주의] {path.name} 이(가) 열려 있어 "
                    f"{target.name} 으로 저장했습니다.")
                log("         (Excel 등에서 닫은 뒤 다시 실행하면 원래 이름으로 저장됩니다)")
            return target
        except PermissionError:
            continue
        except OSError:
            continue
    if not quiet:
        log(f"  [경고] {path.name} 저장 실패 - 파일이 잠겨 있습니다. 계속 진행합니다.")
    return None


# ---------------------------------------------------------------- 1. 환경 점검
def check_env(only_a):
    log("=" * 60)
    log("효소 기능 분류 예비실험 - " + ("모델 A만" if only_a else "모델 A & B"))
    log(f"실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 60)
    log(f"python  : {sys.version.split()[0]}")

    needed = [
        ("numpy", "numpy"), ("pandas", "pandas"), ("requests", "requests"),
        ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"),
    ]
    if not only_a:
        needed += [("torch", "torch"), ("transformers", "transformers")]

    missing = []
    for mod, pip_name in needed:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)

    if missing:
        log("")
        log("[중단] 다음 패키지가 없습니다:")
        log("  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
            + " ".join(missing))
        if "torch" in missing:
            log("")
            log("  torch가 느리면 모델 A만 먼저 돌릴 수 있습니다:")
            log("      python run_prelim.py --only-a")
        save_log()
        sys.exit(1)

    if only_a:
        log("모드    : 모델 A만 (torch 사용 안 함)")
    else:
        import torch
        if torch.cuda.is_available():
            log(f"GPU     : {torch.cuda.get_device_name(0)}")
        else:
            log("GPU     : 없음 (CPU로 진행 - 이 규모는 몇 분이면 끝납니다)")
    log("")


# ---------------------------------------------------------------- 2. 데이터 수집
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"


def fetch_ec_class(ec_digit, n):
    """EC 1단계가 ec_digit인 리뷰된 단백질을 최대 n개 가져온다."""
    import pandas as pd
    import requests

    query = (
        f"(ec:{ec_digit}.*) AND (reviewed:true) "
        f"AND (length:[{MIN_LEN} TO {MAX_LEN}]) AND (fragment:false)"
    )
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,ec,sequence,protein_name",
        "size": min(500, max(n * 4, 25)),
    }
    r = requests.get(UNIPROT, params=params, timeout=120)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    if df.empty:
        return df

    df = df.rename(columns={
        "Entry": "accession",
        "EC number": "ec",
        "Sequence": "sequence",
        "Protein names": "name",
    })
    df = df.dropna(subset=["ec", "sequence"])

    # EC 번호가 하나뿐인 것만 (다중 기능 효소 제외)
    df = df[~df["ec"].astype(str).str.contains(";")]
    # 1단계 숫자 재확인
    df = df[df["ec"].astype(str).str.strip().str.startswith(f"{ec_digit}.")]
    # 비표준 아미노산 글자 제외
    df = df[~df["sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)]

    df = df.drop_duplicates(subset=["sequence"])
    df["label"] = ec_digit
    return df.head(n).reset_index(drop=True)


def collect_data():
    import pandas as pd

    log("[1/5] UniProt에서 서열 수집")
    frames = []
    for ec in EC_CLASSES:
        try:
            sub = fetch_ec_class(ec, N_PER_CLASS)
        except Exception as e:
            log(f"  EC {ec}: 실패 - {type(e).__name__}: {e}")
            continue
        log(f"  EC {ec} ({EC_NAMES[ec]:>14}): {len(sub):3d}개")
        frames.append(sub)
        time.sleep(0.5)          # UniProt 서버 배려

    if not frames:
        log("")
        log("[중단] 서열을 하나도 받지 못했습니다.")
        log("  - 인터넷 연결 또는 회사/학교 방화벽을 확인하세요.")
        log("  - 위 오류 메시지를 Claude에게 그대로 보여주세요.")
        save_log()
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True)
    lens = data["sequence"].str.len()
    log("")
    log(f"  총 {len(data)}개 서열")
    log(f"  길이: 최소 {lens.min()}, 중앙값 {int(lens.median())}, 최대 {lens.max()}")

    counts = data["label"].value_counts().sort_index()
    log("")
    log("  클래스 분포:")
    for ec, c in counts.items():
        log(f"    EC {ec} {EC_NAMES[ec]:>14}: {'#' * min(c, 40)} ({c})")

    if counts.min() < N_FOLDS:
        log("")
        log(f"  [경고] {N_FOLDS}개 미만인 클래스가 있어 교차검증이 불안정합니다.")
        log("         N_PER_CLASS를 늘리거나 해당 클래스를 제외하세요.")

    saved = safe_write(
        OUT_DIR / "sequences.csv",
        lambda p: data.to_csv(p, index=False, encoding="utf-8"),
    )
    if saved:
        log(f"  저장: {saved.name}")
    log("")
    return data


# ---------------------------------------------------------------- 3. 모델 A
def composition_features(seq):
    """20차원 아미노산 빈도 벡터. 순서 정보는 버린다."""
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
    log("[2/5] 모델 A - 아미노산 조성")
    X = np.vstack([composition_features(s) for s in data["sequence"]])
    assert np.allclose(X.sum(axis=1), 1.0), "조성 벡터의 합은 1이어야 함"
    log(f"  특징 행렬: {X.shape}  (행 합계 검증 통과)")
    log("")
    return X


# ---------------------------------------------------------------- 4. 모델 B
def build_model_b(data):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    log("[3/5] 모델 B - ESM-2 8M 임베딩")
    log(f"  모델 다운로드/로드: {MODEL_NAME}")
    log("  (처음 실행 시 약 30MB를 받습니다)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    esm = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    n_params = sum(p.numel() for p in esm.parameters())
    log(f"  로드 완료: {n_params/1e6:.1f}M 파라미터, {device}")
    log("  (ESM-2는 학습하지 않고 특징 추출기로만 사용)")

    sequences = data["sequence"].tolist()
    batch_size = 8
    out = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LEN + 2)
            enc = {k: v.to(device) for k, v in enc.items()}

            hidden = esm(**enc).last_hidden_state       # (B, L, 320)
            mask = enc["attention_mask"].clone()        # (B, L)

            # 시작(cls) 토큰 제외
            mask[:, 0] = 0
            # 종료(eos) 토큰 제외 - 각 서열의 마지막 유효 위치
            lengths = enc["attention_mask"].sum(dim=1)
            mask[torch.arange(mask.size(0), device=device), lengths - 1] = 0

            m = mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            out.append(pooled.cpu().numpy())

            done = min(i + batch_size, len(sequences))
            print(f"\r  임베딩 {done}/{len(sequences)}", end="", flush=True)

    print()
    X = np.vstack(out)
    log(f"  특징 행렬: {X.shape}")
    log("")
    return X


# ---------------------------------------------------------------- 5. 평가
def evaluate(X_A, X_B, y):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    log("[4/5] 평가 (5-fold 교차검증, 무작위 분할)")

    def make_clf():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
        )

    n_splits = min(N_FOLDS, int(np.bincount(y)[np.bincount(y) > 0].min()))
    if n_splits < N_FOLDS:
        log(f"  [조정] 최소 클래스 크기 때문에 {n_splits}-fold로 진행")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_SEED)

    models = [("A: 조성 (20d)", X_A)]
    if X_B is not None:
        models.append(("B: ESM-2 8M (320d)", X_B))

    results, preds = {}, {}
    for name, X in models:
        p = cross_val_predict(make_clf(), X, y, cv=cv)
        preds[name] = p
        results[name] = {
            "accuracy": float(accuracy_score(y, p)),
            "macro_F1": float(f1_score(y, p, average="macro", zero_division=0)),
        }

    # 라벨 셔플 대조군 - 파이프라인 누수 점검
    X_ctrl = X_B if X_B is not None else X_A
    y_shuf = np.random.RandomState(0).permutation(y)
    p_shuf = cross_val_predict(make_clf(), X_ctrl, y_shuf, cv=cv)
    results["(대조군) 라벨 셔플"] = {
        "accuracy": float(accuracy_score(y_shuf, p_shuf)),
        "macro_F1": float(f1_score(y_shuf, p_shuf, average="macro",
                                   zero_division=0)),
    }

    chance = 1.0 / len(np.unique(y))
    log("")
    log(f"  {'모델':<26} {'정확도':>8} {'macro-F1':>10}")
    log("  " + "-" * 46)
    for k, v in results.items():
        log(f"  {k:<26} {v['accuracy']:>8.3f} {v['macro_F1']:>10.3f}")
    log(f"\n  무작위 추측 수준 = {chance:.3f}")

    # 자동 해석
    log("")
    log("  [점검]")
    f1_shuf = results["(대조군) 라벨 셔플"]["macro_F1"]
    if f1_shuf > chance + 0.10:
        log(f"  X 셔플 대조군이 {f1_shuf:.3f}로 높습니다. 코드 누수 의심.")
    else:
        log(f"  O 셔플 대조군 {f1_shuf:.3f} - 정상 (누수 없음)")

    f1_a = results["A: 조성 (20d)"]["macro_F1"]
    if X_B is not None:
        f1_b = results["B: ESM-2 8M (320d)"]["macro_F1"]
        if f1_b > f1_a + 0.03:
            log(f"  O B가 A보다 {f1_b - f1_a:+.3f} 높음 - 언어모델이 기여함")
        else:
            log(f"  ? B와 A 차이가 {f1_b - f1_a:+.3f}뿐. 데이터를 늘려보세요.")
    else:
        log("  - 모델 B는 실행하지 않음 (--only-a)")

    main_name = "B: ESM-2 8M (320d)" if X_B is not None else "A: 조성 (20d)"
    log("")
    log(f"[5/5] 클래스별 성능 ({main_name})")
    labels = sorted(np.unique(y))
    target_names = [f"EC{e} {EC_NAMES[e]}" for e in labels]
    report = classification_report(y, preds[main_name],
                                   labels=labels, target_names=target_names,
                                   digits=3, zero_division=0)
    log(report)

    # 혼동 행렬 그림
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 그림 라벨은 ASCII로 - 한글 폰트가 없는 환경에서 네모로 깨지는 것 방지
        ascii_title = {
            "A: 조성 (20d)": "A: AA composition (20d)",
            "B: ESM-2 8M (320d)": "B: ESM-2 8M (320d)",
        }
        names = [n for n, _ in models]
        fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 5),
                                 squeeze=False)
        axes = axes[0]
        for ax, name in zip(axes, names):
            cm = confusion_matrix(y, preds[name], labels=labels,
                                  normalize="true")
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"EC{l}" for l in labels])
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels([f"EC{l}" for l in labels])
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{ascii_title.get(name, name)}\n"
                         f"macro-F1 = {results[name]['macro_F1']:.3f}")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if cm[i, j] > 0.01:
                        ax.text(j, i, f"{cm[i,j]:.2f}", ha="center",
                                va="center", fontsize=8,
                                color="white" if cm[i, j] > 0.5 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        saved = safe_write(
            OUT_DIR / "confusion_prelim.png",
            lambda p: plt.savefig(p, dpi=150, bbox_inches="tight"),
        )
        plt.close()
        if saved:
            log(f"  그림 저장: {saved.name}")
    except Exception as e:
        log(f"  그림 생성 실패 (결과에는 영향 없음): {e}")

    return results


# ---------------------------------------------------------------- main
def main():
    import numpy as np

    only_a = "--only-a" in sys.argv

    np.random.seed(RANDOM_SEED)
    check_env(only_a)

    data = collect_data()
    X_A = build_model_a(data)
    X_B = None if only_a else build_model_b(data)
    y = data["label"].to_numpy()

    # 특징 저장 - 재계산 없이 분류기 실험을 반복할 수 있다
    arrays = {"X_A": X_A, "y": y, "accession": data["accession"].to_numpy()}
    if X_B is not None:
        arrays["X_B"] = X_B
    saved = safe_write(OUT_DIR / "features_prelim.npz",
                       lambda p: np.savez_compressed(p, **arrays))
    if saved:
        log(f"  저장: {saved.name}  (다시 쓸 때는 ESM-2 재실행 불필요)")
    log("")

    results = evaluate(X_A, X_B, y)

    payload = json.dumps({
        "timestamp": datetime.now().isoformat(),
        "mode": "only-a" if only_a else "A+B",
        "settings": {
            "n_per_class": N_PER_CLASS,
            "min_len": MIN_LEN,
            "max_len": MAX_LEN,
            "seed": RANDOM_SEED,
            "n_sequences": int(len(data)),
        },
        "results": results,
    }, ensure_ascii=False, indent=2)
    safe_write(OUT_DIR / "results_prelim.json",
               lambda p: p.write_text(payload, encoding="utf-8"))

    log("=" * 60)
    log("완료. 생성된 파일:")
    log("  sequences.csv         수집한 서열")
    log("  features_prelim.npz   특징 벡터 (재사용용)")
    log("  results_prelim.json   성능 수치")
    log("  confusion_prelim.png  혼동 행렬")
    log("  run_log.txt           이 로그 전체")
    if only_a:
        log("")
        log("torch 설치가 끝나면 모델 B까지:")
        log("  python run_prelim.py")
    log("=" * 60)
    save_log()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        log("")
        log("[오류 발생]")
        log(tb)
        save_log()
        print("\nrun_log.txt 를 Claude에게 보여주세요.")
        sys.exit(1)
