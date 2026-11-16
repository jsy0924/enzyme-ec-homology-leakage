# BLAST 비교 실험 (모델 D)

## 무엇을 하나

테스트 서열마다 훈련 서열 중 **정렬로 가장 닮은 것**을 찾아 그 EC 번호를 그대로 복사합니다. 기준(e-value ≤ 0.001)을 넘는 훈련 서열이 없으면 답을 내지 않고, 오답으로 셉니다.

분할은 `run\_main.py`의 `build\_splits()`를 그대로 불러 쓰므로, 모델 A·B·C와 **완전히 같은 훈련·테스트 구성**(5가지 분할 × 5 fold, 시드 42)에서 평가됩니다.

## 파일

|파일|내용|
|-|-|
|`01\_make\_fasta.py`|`../sequences\_main.csv` → `all.fasta`|
|`02\_run\_diamond.bat`|DIAMOND 전체 정렬 (선택 사항, `diamond.exe` 필요)|
|`hits.tsv`|정렬 결과 (DIAMOND 2.1.10, 미리 계산해 둠)|
|`03\_blast\_predict.py`|핵심 예측 코드|
|`run\_blast.bat`|03번을 더블클릭으로 실행|

## 실행

`run\_blast.bat`을 더블클릭하면 됩니다. `hits.tsv`가 들어 있으니 DIAMOND는 필요 없습니다. 분할을 다시 만드느라 1\~2분 걸립니다.

결과:

* `blast\_predictions.csv` — 서열마다, 분할마다 정답 / 예측 / 가장 닮은 훈련 서열과의 동일성(%)
* `blast\_summary.txt` — 분할별 macro-F1, 답을 낸 비율, 답했을 때 정확도

## 정렬 설정 (보고서 방법론용)

```
diamond makedb --in all.fasta -d all
diamond blastp -d all -q all.fasta -o hits.tsv --very-sensitive --evalue 1e-3
               --max-target-seqs 0
               --outfmt 6 qseqid sseqid pident length evalue bitscore qcovhsp scovhsp
```

|설정|값|의미|
|-|-|-|
|도구|DIAMOND 2.1.10 blastp|BLAST와 같은 방식의 단백질 정렬, 더 빠름|
|민감도|`--very-sensitive`|먼 친척까지 찾기|
|e-value|≤ 1e-3|우연히 닮았을 가능성 0.001 이하만 인정|
|대상 수|제한 없음|찾은 상대를 전부 기록|
|순위|bit score|데이터베이스 크기와 무관한 점수|

## 참고

* 7,000개를 한 번만 전부 정렬하고, fold마다 결과에서 **훈련 서열만 남겨** 썼습니다. fold마다 따로 검색하는 것과 같은 결과이고, bit score는 데이터베이스 크기와 무관합니다. 다만 e-value는 7,000개 기준으로 계산되어, fold별 데이터베이스(5,600개) 기준보다 약간 엄격합니다.
* e-value 기준을 10으로 느슨하게 해도 가장 엄격한 분할의 macro-F1은 0.838로, 1e-3일 때(0.827)와 결론이 같았습니다.
* 정렬은 Linux에서 계산했습니다.

