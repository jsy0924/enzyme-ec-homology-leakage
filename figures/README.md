# 논문에 사용한 그림

| 파일 | 논문 번호 | 내용 | 생성 |
|---|---|---|---|
| `fig_split.png` | 그림 1 | 무작위 분할과 클러스터 분할의 차이 (개념도) | `sources/split_diagram.svg` |
| `fig_pipeline.png` | 그림 2 | 실험 파이프라인 (개념도) | `sources/fig_pipeline.svg` |
| `fig_main.png` | 그림 3 | 분할별 macro-F1, 하락폭, 동일성 구간별 정확도 | `../03_blast/analysis/make_fig5.py` |
| `fig_perclass_drop.png` | 그림 4 | EC 클래스별 F1 하락폭 | `make_figs.py` |
| `fig_confusion.png` | 그림 5 | 가장 엄격한 분할의 혼동 행렬 (A, B, C) | `../02_stats/make_confusions.py` |
| `fig_identity_ec.png` | 그림 6 | 서열 동일성과 EC 번호 일치 수준 | `make_figs.py` |
| `fig_weights.png` | 그림 7 | 아미노산 조성 모델의 클래스별 가중치 | `make_fig_weights.py` |

개념도 두 개는 SVG로 직접 작성한 뒤 PNG로 변환했습니다. `sources/`에 SVG 원본이 있습니다.
데이터 그래프는 축과 범례가 영어입니다.
