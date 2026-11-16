# 본 실험

## run\_main.py의 단계

1. UniProt REST API에서 EC 클래스별 1,000개씩 서열 수집 (클래스별 캐시, 실패 시 재시도)
2. ESM-2 임베딩 계산 (평균 풀링, 시작·종료 토큰과 패딩 제외) → `emb\_\*\_7000.npz`
3. 4-mer Jaccard 유사도로 서열 가족 묶기 (연결 요소) → 클러스터 분할 4단계
4. 훈련–테스트 간 최대 유사도 재계산으로 누출 검증
5. 모델 A·B·C 학습 및 평가 (5 분할 × 5 fold × 3 모델 = 75회)
6. 결과 JSON과 그림 출력

## 주요 옵션

|옵션|뜻|
|-|-|
|`--n 1000`|EC 클래스별 서열 수|
|`--esm 8M,650M`|사용할 ESM-2 모델|
|`--refetch`|UniProt에서 서열을 다시 받기|
|`--reembed`|임베딩을 다시 계산|
|`--only-a`|조성 기준선만 실행|

## 실행 기록

* `main\_log.txt` — 본 실험 실행 로그 (분할별 수치, 누출 검증 결과 포함)
* `results\_main.json` — 분할별·모델별 macro-F1과 클래스별 지표
* `run\_log.txt`, `results\_prelim.json` — 예비 실험(2,000개 서열, 모델 A·B) 기록

## .bat 파일을 쓴 이유

Windows에서 파이썬 경로가 다른 프로그램(Inkscape)의 파이썬으로 잡히는 문제가 있어,
`%LOCALAPPDATA%\\Programs\\Python\\Python311\\python.exe`와 `py -3`을 순서대로 찾도록 했습니다.
한글이 포함되면 배치 파일 파싱이 깨지므로 ASCII로만 작성했습니다.

