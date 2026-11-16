"""
1단계: sequences_main.csv 의 7,000개 서열을 FASTA 파일로 저장한다.

FASTA는 서열 분석 도구(BLAST, DIAMOND 등)가 공통으로 읽는 형식이다.
    >서열이름
    MKTAYIAKQRQ...

실행:  (프로젝트 폴더의 blast 폴더 안에서)
    py -3 01_make_fasta.py

결과:  all.fasta
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def main():
    df = pd.read_csv(PROJECT / "sequences_main.csv")
    assert df["accession"].is_unique, "서열 이름(accession)이 중복됨"
    assert df["sequence"].is_unique, "같은 서열이 두 번 들어 있음"

    out = HERE / "all.fasta"
    with open(out, "w", encoding="ascii", newline="\n") as f:
        for acc, seq in zip(df["accession"], df["sequence"]):
            f.write(f">{acc}\n{seq}\n")
    print(f"{len(df)}개 서열 -> {out.name}")


if __name__ == "__main__":
    main()
