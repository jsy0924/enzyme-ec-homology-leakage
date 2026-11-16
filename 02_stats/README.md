# 통계 검정

`01\_main/run\_main.py`를 수정하지 않고, 저장된 데이터와 임베딩을 다시 읽어 예측값을
재생성한 뒤 불확실성을 계산합니다.

## 실행 순서

|파일|하는 일|출력|
|-|-|-|
|`reproduce.py`|다섯 분할 × 세 모델의 예측값을 재생성|`preds.npz`|
|`bootstrap.py`|서열 가족 단위 부트스트랩 2,000회 (시드 2026)|`bootstrap\_results.json`|
|`seeds.py`|fold 배정 시드 10종으로 교차검증 반복|`seeds\_results.json`, `seeds\_log.txt`|
|`summarize.py`|표로 정리|`stats\_table.csv`, `stats\_results.json`|
|`make\_confusions.py`|모델별 혼동 행렬 (덮어쓰기 방지)|`outputs/fig2\_confusion\_\*.png`|
|`make\_stats\_fig.py`|신뢰구간 그림|`outputs/fig4\_statistics.png`|

## 왜 가족 단위 부트스트랩인가

같은 서열 가족에 속한 서열은 서로 독립이 아닙니다. 개별 서열을 재표본추출하면 신뢰구간이
실제보다 좁아집니다. 그래서 가장 엄격한 분할의 클러스터 3,580개를 재표본추출 단위로 삼았습니다.
두 조건의 차이는 같은 재표본에서 각각 계산한 뒤 뺍니다.

## 경로

스크립트 상단의 경로 상수를 자신의 환경에 맞게 바꿔야 합니다. Linux 환경에서 실행한 코드입니다.

