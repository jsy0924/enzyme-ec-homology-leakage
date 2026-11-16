"""
3단계: BLAST 방식(최근접 이웃 라벨 복사)으로 EC 클래스를 예측한다.  = 모델 D

방법
  테스트 서열마다, 그 fold의 "훈련 서열" 중에서 정렬 점수(bit score)가 가장
  높은 서열을 찾아 그 EC 번호를 그대로 복사한다.
  기준(e-value <= 0.001)을 넘는 훈련 서열이 하나도 없으면 답을 내지 않는다(0).
  답 없음(0)은 채점할 때 오답으로 센다.

분할
  run_main.py 의 build_splits() 를 그대로 불러서, 모델 A·B·C와 완전히 같은
  5가지 분할 x 5개 fold 배정(시드 42)을 쓴다.

정렬 결과
  hits.tsv : DIAMOND 2.1.10 으로 7,000개끼리 전부 정렬한 결과 (이미 들어 있음).
             02_run_diamond.bat 으로 다시 만들 수도 있다.

실행:  run_blast.bat 더블클릭  (또는  py -3 03_blast_predict.py)

결과
  blast_predictions.csv : 서열마다, 분할마다 정답 / BLAST 예측 / 가장 닮은 훈련 서열과의 동일성
  blast_summary.txt     : 분할별 macro-F1, 답을 낸 비율, 답했을 때 정확도
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

import run_main as R  # noqa: E402  (원래 코드는 불러서 쓰기만 하고 고치지 않는다)

LABELS = [1, 2, 3, 4, 5, 6, 7]


def load_hits(acc_to_idx):
    """정렬 결과를 '서열 번호 -> 닮은 서열 목록(점수 높은 순)' 으로 정리한다."""
    hits = pd.read_csv(HERE / "hits.tsv", sep="\t", header=None,
                       names=["q", "s", "pident", "length", "evalue",
                              "bitscore", "qcov", "scov"])
    hits = hits[hits.q != hits.s]                    # 자기 자신과의 정렬 제외
    hits["qi"] = hits.q.map(acc_to_idx)
    hits["si"] = hits.s.map(acc_to_idx)
    if hits[["qi", "si"]].isna().any().any():
        raise SystemExit("hits.tsv 의 서열 이름이 sequences_main.csv 와 맞지 않습니다.")
    # 질의마다 bit score 높은 순. 점수가 같으면 번호 작은 쪽을 먼저 (결과 고정)
    hits = hits.sort_values(["qi", "bitscore", "si"], ascending=[True, False, True])

    targets = {}
    for qi, g in hits.groupby("qi", sort=False):
        targets[int(qi)] = (g.si.to_numpy(dtype=np.int64), g.pident.to_numpy())
    return targets


def predict(y, fold_of, targets):
    """fold 배정을 받아 모든 서열의 BLAST 예측을 만든다.

    y        : 정답 EC 번호 (1~7)
    fold_of  : 각 서열이 몇 번 더미에 있는지 (0~4)
    반환     : pred (0 = 답 없음), ident (가장 닮은 훈련 서열과의 동일성 %, 없으면 nan)
    """
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    ident = np.full(n, np.nan)
    empty = (np.empty(0, dtype=np.int64), np.empty(0))

    for k in np.unique(fold_of):                    # 더미 0~4 를 차례로 테스트
        in_train = fold_of != k                     # 나머지 4더미 = 훈련
        for i in np.where(fold_of == k)[0]:         # 테스트 서열 하나씩
            si, pid = targets.get(int(i), empty)    # 닮은 순 목록
            ok = in_train[si]                       # 그중 훈련 더미에 있는 것만
            if ok.any():
                j = int(np.argmax(ok))              # 첫 번째 = 가장 닮은 훈련 서열
                pred[i] = y[si[j]]                  # 그 EC 번호를 복사
                ident[i] = pid[j]
    return pred, ident


def main():
    R.log = lambda *a, **k: None                    # run_main 의 로그 출력은 끔
    data = pd.read_csv(PROJECT / "sequences_main.csv")
    y = data["label"].to_numpy()
    acc_to_idx = {a: i for i, a in enumerate(data["accession"])}

    print("정렬 결과 불러오는 중 ...")
    targets = load_hits(acc_to_idx)
    print(f"  닮은 서열이 하나 이상 있는 서열: {len(targets)} / {len(y)}")

    print("분할 재구성 중 (run_main.build_splits, 1분 정도) ...")
    splits = R.build_splits(data, y)

    out = pd.DataFrame({"accession": data["accession"], "true_EC": y})
    lines = ["BLAST(DIAMOND) 최근접 이웃 예측 - 모델 D",
             "e-value <= 1e-3, 가장 높은 bit score 의 훈련 서열 EC 복사, 답 없음 = 오답",
             "",
             f"{'분할':<14}{'macro-F1':>10}{'답한 비율':>11}{'답했을 때 정확도':>17}"
             f"{'동일성 중앙값':>14}{'동일성 최대':>12}"]

    for name, s in splits.items():
        fold_of = np.empty(len(y), dtype=np.int64)
        for k, (_, te) in enumerate(s["folds"]):
            fold_of[te] = k
        pred, ident = predict(y, fold_of, targets)

        f1 = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
        answered = pred > 0
        acc = (pred[answered] == y[answered]).mean()
        v = ident[~np.isnan(ident)]
        lines.append(f"{name:<14}{f1:>10.3f}{answered.mean():>11.1%}{acc:>17.1%}"
                     f"{np.median(v):>13.1f}%{v.max():>11.1f}%")

        out[f"pred_{name}"] = pred
        out[f"identity_{name}"] = np.round(ident, 1)

    out.to_csv(HERE / "blast_predictions.csv", index=False, encoding="utf-8")
    text = "\n".join(lines)
    (HERE / "blast_summary.txt").write_text(text + "\n", encoding="utf-8")
    print()
    print(text)
    print("\n저장: blast_predictions.csv, blast_summary.txt")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
