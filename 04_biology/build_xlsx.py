"""Workbook for the shared survey sheet: guide + 3 case tabs + summary."""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

OUT = Path("/home/claude/bio/out")
HEAD = PatternFill("solid", fgColor="F3F3F3")
WRAP = Alignment(wrap_text=True, vertical="top")
TOP = Alignment(vertical="top")
EC = {1: "산화환원효소", 2: "전이효소", 3: "가수분해효소", 4: "분해효소",
      5: "이성질화효소", 6: "연결효소", 7: "운반효소"}
CAT = {"A_BLAST no hit, 650M right": "A. BLAST 답 없음 · 650M 맞힘",
       "B_BLAST no hit, 650M wrong": "B. BLAST 답 없음 · 650M 틀림",
       "C_BLAST right, 650M wrong": "C. BLAST 맞힘 · 650M 틀림",
       "D_BLAST wrong, 650M right": "D. BLAST 틀림 · 650M 맞힘"}


def uni(a):
    return f"https://www.uniprot.org/uniprotkb/{a}/entry"


def clean(t):
    return " ".join(str(t).split())


def header(ws, names, widths):
    ws.append(names)
    for i, w in enumerate(widths, start=1):
        c = ws.cell(row=1, column=i)
        c.fill = HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[c.column_letter].width = w


def link(cell, acc):
    cell.value = acc
    cell.hyperlink = uni(acc)


wb = Workbook()

# ---------------- 안내 ----------------
g = wb.active
g.title = "안내"
header(g, ["항목", "내용"], [22, 100])
guide = [
    ("이 표는", "ESM-2 650M과 BLAST의 예측이 갈린 단백질, 그리고 서열은 닮았는데 기능이 다른 단백질을 생물학적으로 조사하기 위한 표입니다. 노란 칸이 채울 칸입니다."),
    ("데이터 기준", "Swiss-Prot 효소 7,000개(EC 클래스 7개 × 1,000개, UniProt 2026_04). 1번 시트는 가장 엄격한 분할(4-mer Jaccard < 0.05)의 예측입니다. BLAST는 DIAMOND 2.1.10, e-value ≤ 0.001로 훈련 서열 중 가장 닮은 것의 EC를 복사한 결과입니다."),
    ("표본 규칙", "사례는 미리 정한 규칙으로 뽑았습니다(시드 2026). 1번: 범주 4개 × EC 클래스 7개 × 2건 = 56건. 2번 고동일성: 해당하는 37쌍 전부. 2번 부분 정렬: 62쌍 중 20쌍 무작위. 사례를 빼거나 바꾸거나 더 고르지 마세요. 결론이 '이야기가 잘 되는 사례만 고른 결과'가 아님을 보장하는 장치입니다."),
    ("조사 순서", "1번 시트의 '1차' 28건을 먼저 하고, 그다음 '2차' 28건, 그다음 2번 시트들을 합니다."),
    ("범주 A", "BLAST 답 없음 · 650M 맞힘. 훈련 서열 중 닮은 것이 없어 BLAST는 답을 못 냈는데 650M이 맞힌 경우. 전체 984건."),
    ("범주 B", "BLAST 답 없음 · 650M 틀림. 전체 523건."),
    ("범주 C", "BLAST 맞힘 · 650M 틀림. 언어모델이 실패한 경우. 범주 A만큼 중요합니다. 전체 342건."),
    ("범주 D", "BLAST 틀림 · 650M 맞힘. BLAST가 닮은 서열을 찾았지만 기능이 달랐던 경우. 전체 123건."),
    ("'데이터셋 내 고아'", "'예'는 7,000개 전체에서 닮은 서열(e ≤ 0.001)이 하나도 없는 서열입니다(전체 556개)."),
    ("① 외부 BLAST", "NCBI BLAST(blastp)에서 UniProtKB/Swiss-Prot 또는 nr 전체를 대상으로 검색합니다. 자기 자신을 뺀 최상위 결과의 이름과 동일성(%)을 적고, 상동체가 있는지 고릅니다. 범주 A·B에서 가장 중요합니다. 우리 데이터에서는 고아여도 전체 데이터베이스에는 친척이 있을 수 있고, ESM-2는 사전훈련 때 그 친척들을 봤을 수 있습니다. 친척이 전혀 없는데 650M이 맞힌 사례가 가장 강한 사례입니다."),
    ("② 기능 요약", "UniProt 항목의 Function과 Catalytic activity를 한두 문장으로."),
    ("③ 도메인 구성", "InterPro 또는 Pfam 도메인을 N말단부터 순서대로. 예: Pkinase → SH2."),
    ("④ 활성 부위 주석", "UniProt 항목 'Features'에 Active site 또는 Binding site 주석이 있는지."),
    ("⑤ 틀린 예측의 타당성", "틀린 모델이 고른 클래스가 생물학적으로 그럴듯한지. 예: 비슷한 반응, 비슷한 기질, 다기능 효소."),
    ("⑥ 해석 범주 · 확신도", "드롭다운에서 고릅니다. 맞는 것이 없으면 '설명 못 함'을 고르고 메모에 이유를 적습니다."),
    ("2번 고동일성", "서열 거의 전체(양쪽 80% 이상)가 동일성 50% 이상으로 정렬되는데 EC 클래스가 다른 쌍입니다. 두 단백질의 기능과 도메인 구성을 비교하고, 왜 다른지 고릅니다. EC7(운반효소)은 2018년에 새로 생겨 다른 클래스에서 옮겨 온 효소가 있으니, 기능이 정말 다른지 주석 시점 차이인지 확인하세요. 37쌍 중 EC6 ↔ EC7이 10쌍으로 가장 많습니다."),
    ("2번 부분 정렬", "동일성은 50% 이상인데 한쪽 서열의 50% 미만만 정렬된 쌍입니다. 공유하는 도메인 하나 때문에 BLAST가 기능을 잘못 옮길 수 있는 경우입니다. 정렬된 구간이 어느 도메인인지 확인하세요."),
    ("기능 비교 기준", "기능은 EC 4자리까지 비교하세요. '2 요약' 시트를 보면 동일성 30% 미만에서는 EC 1자리가 같은 쌍이 96%지만 4자리까지 같은 쌍은 23%뿐입니다."),
    ("쓰면 안 되는 말", "'ESM-2가 이 도메인을 인식해서 맞혔다'처럼 모델의 이유를 단정하지 마세요. 확인할 수 있는 것은 '이 단백질에는 이런 특징이 있고, 이는 정답 클래스와 일치한다'까지입니다."),
    ("도구: UniProt", "https://www.uniprot.org"),
    ("도구: InterPro", "https://www.ebi.ac.uk/interpro"),
    ("도구: AlphaFold DB", "https://alphafold.ebi.ac.uk"),
    ("도구: NCBI BLAST", "https://blast.ncbi.nlm.nih.gov/Blast.cgi"),
    ("전체 목록", "표본에 뽑히지 않은 사례까지 전부: 프로젝트 폴더 enzyme-project\\biology 의 CSV 파일."),
]
for k, v in guide:
    g.append([k, v])
    r = g.max_row
    g.cell(row=r, column=1).alignment = TOP
    g.cell(row=r, column=1).font = Font(bold=True)
    g.cell(row=r, column=2).alignment = WRAP
    if v.startswith("https://"):
        g.cell(row=r, column=2).hyperlink = v
g.freeze_panes = "A2"

# ---------------- 1 예측차이 ----------------
s = pd.read_csv(OUT / "idea1_sample.csv")
w1 = wb.create_sheet("1 예측차이")
names1 = ["사례", "UniProt", "단백질 이름", "범주", "정답", "650M 예측", "BLAST 예측",
          "데이터셋 내 고아", "전체 EC", "생물종", "BLAST 최근접 훈련 서열", "최근접 EC",
          "최근접 동일성(%)", "차수",
          "외부 BLAST", "외부 최상위 서열 · 동일성", "기능 요약", "도메인 구성",
          "활성 부위 주석", "틀린 예측의 타당성", "해석 범주", "확신도", "메모", "조사자"]
widths1 = [7, 10, 34, 26, 16, 10, 10, 9, 11, 22, 34, 11, 10, 6,
           13, 26, 40, 30, 11, 30, 22, 8, 34, 10]
header(w1, names1, widths1)
for r in s.itertuples():
    near = "" if pd.isna(r.BLAST_nearest_identity_pct) else \
        f"{r.BLAST_nearest_train_accession} · {clean(r.BLAST_nearest_train_name)}"
    w1.append([r.case_id, None, clean(r.protein_name), CAT[r.category],
               f"EC{r.true_class} {EC[r.true_class]}", f"EC{r.pred_650M}",
               "답 없음" if r.pred_BLAST == 0 else f"EC{r.pred_BLAST}",
               "예" if r.orphan_in_dataset == "yes" else "아니오",
               str(r.EC_full), clean(r.organism), near or None,
               str(r.BLAST_nearest_train_EC) if near else None,
               None if pd.isna(r.BLAST_nearest_identity_pct) else float(r.BLAST_nearest_identity_pct),
               f"{r.round}차"])
    row = w1.max_row
    link(w1.cell(row=row, column=2), r.accession)
    for col in range(1, 25):
        w1.cell(row=row, column=col).alignment = WRAP if col in (3, 10, 11, 16, 17, 18, 20, 23) else TOP
    w1.cell(row=row, column=9).number_format = "@"
    w1.cell(row=row, column=12).number_format = "@"
    w1.cell(row=row, column=13).number_format = "0.0"
w1.freeze_panes = "D2"


# ---------------- 2 pairs ----------------
def pair_sheet(title, csv):
    t = pd.read_csv(OUT / csv)
    ws = wb.create_sheet(title)
    names = ["쌍", "서열 1", "이름 1", "EC 1", "서열 2", "이름 2", "EC 2", "동일성(%)",
             "정렬 범위 1(%)", "정렬 범위 2(%)", "생물종 1", "생물종 2",
             "기능이 실제로 다른가", "도메인 구성", "설명 범주", "근거", "메모", "조사자"]
    widths = [7, 10, 32, 10, 10, 32, 10, 9, 9, 9, 20, 20, 16, 13, 22, 40, 30, 10]
    header(ws, names, widths)
    for r in t.itertuples():
        ws.append([r.pair_id, None, clean(r.name_1), str(r.EC_1), None, clean(r.name_2),
                   str(r.EC_2), float(r.identity_pct), float(r.coverage_1_pct),
                   float(r.coverage_2_pct), clean(r.organism_1), clean(r.organism_2)])
        row = ws.max_row
        link(ws.cell(row=row, column=2), r.accession_1)
        link(ws.cell(row=row, column=5), r.accession_2)
        for col in range(1, 19):
            ws.cell(row=row, column=col).alignment = WRAP if col in (3, 6, 11, 12, 16, 17) else TOP
        for col in (4, 7):
            ws.cell(row=row, column=col).number_format = "@"
        for col in (8, 9, 10):
            ws.cell(row=row, column=col).number_format = "0.0"
    ws.freeze_panes = "B2"
    return len(t)


n2h = pair_sheet("2 고동일성", "idea2_high_identity_different_class.csv")
n2p = pair_sheet("2 부분정렬", "idea2_partial_match_sample.csv")

# ---------------- 2 요약 ----------------
sm = pd.read_csv(OUT / "idea2_identity_vs_EC_level.csv")
w4 = wb.create_sheet("2 요약")
header(w4, ["동일성 구간", "쌍 수", "EC 1자리 같음", "EC 2자리 같음", "EC 3자리 같음",
            "EC 4자리 같음"], [14, 10, 14, 14, 14, 14])
for r in sm.itertuples():
    w4.append([r.identity_bin, int(r.n_pairs), r.same_EC_level1, r.same_EC_level2,
               r.same_EC_level3, r.same_EC_level4])
    row = w4.max_row
    w4.cell(row=row, column=2).number_format = "#,##0"
    for col in range(3, 7):
        w4.cell(row=row, column=col).number_format = "0.0%"
w4.append([])
w4.append(["서열 전체(양쪽 80% 이상)가 정렬된 쌍만 셌습니다. 7,000개 서열의 DIAMOND 전체 정렬(e ≤ 0.001) 기준. "
           "'EC 4자리 같음'은 두 단백질의 EC 번호가 네 자리 모두 같은 비율입니다."])

wb.save(OUT / "survey.xlsx")
print("saved survey.xlsx", len(s), n2h, n2p, len(sm))
