# -*- coding: utf-8 -*-
"""
역량 진단 · 교육 추천 대시보드 (Streamlit)
------------------------------------------------------------
직무를 고르면 → 그 직무의 역량 갭 + 소속 구성원별 교육 추천을 한 화면에.

실행:  streamlit run app.py
필요 파일(같은 폴더):
  · 역량사전_LDA결과.xlsx          (직무x역량_분포)
  · 직원_역량프로파일_46직무.csv
  · 교육콘텐츠_역량태깅.csv
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="역량 진단·교육 추천", layout="wide")

COMP_ORDER = ["사업관리", "고객관리", "품질생산관리", "AI/SW", "해외영업", "시험평가",
              "제어", "HR_AX", "열관리개발", "재무회계", "성능HW"]


# ============================================================
# 데이터 로드 & 전처리 (캐시)
# ============================================================
@st.cache_data
def load_data():
    M = pd.read_excel("역량사전_LDA결과.xlsx", sheet_name="직무x역량_분포", index_col=0)
    M = M[[c for c in M.columns if c.startswith("역량후보")]]
    M.columns = COMP_ORDER
    job_base = (M * 5)                      # 직무기본(요구) 0~5

    emp = pd.read_csv("직원_역량프로파일_46직무.csv", encoding="utf-8-sig")
    tags = pd.read_csv("교육콘텐츠_역량태깅.csv", index_col=0)[COMP_ORDER]
    return job_base, emp, tags


def mm05(df):
    lo, hi = df.values.min(), df.values.max()
    return (df - lo) / (hi - lo) * 5 if hi > lo else df * 0


job_base, emp, tags = load_data()
요구_all = mm05(job_base).round(2)                                  # 직무 × 역량 (0~5)
보유_byjob = emp.groupby("직무")[[f"역량_{c}" for c in COMP_ORDER]].mean()
보유_byjob.columns = COMP_ORDER
보유_all = mm05(보유_byjob).round(2)

JOBS = list(요구_all.index)

# ============================================================
# 사이드바
# ============================================================
st.sidebar.header("⚙️ 설정")
job = st.sidebar.selectbox("직무 선택", JOBS, index=JOBS.index("HRM_채용인사관리")
                           if "HRM_채용인사관리" in JOBS else 0)
top_n = st.sidebar.slider("1인당 추천 강의 수", 1, 5, 3)
st.sidebar.caption("LDA 토픽 기반 요구역량 · 가상 직원 데이터 · 콘텐츠 기반 추천")

st.title("📊 직무 역량 진단 · 교육 추천 대시보드")
st.markdown(f"### 선택 직무 — **{job}**")

# ============================================================
# 1. 직무 요구 vs 보유 (그룹 막대)
# ============================================================
st.subheader("1. 직무 요구역량 vs 보유역량(평균)")
cmp_df = pd.DataFrame({"요구역량": 요구_all.loc[job], "보유역량(평균)": 보유_all.loc[job]})
st.bar_chart(cmp_df, height=300)

# ============================================================
# 2. 구성원 × 역량 갭 히트맵
# ============================================================
st.subheader("2. 구성원별 역량 갭 (요구 − 보유)  · 양수=부족")
members = emp[emp["직무"] == job].copy()
have_cols = [f"역량_{c}" for c in COMP_ORDER]
have_raw = members[have_cols].copy()
have_raw.columns = COMP_ORDER
have_05 = have_raw / have_raw.values.max() * 5 if have_raw.values.max() > 0 else have_raw

req_vec = 요구_all.loc[job]
gap = (req_vec.values[None, :] - have_05.values)
gap_df = pd.DataFrame(gap.round(2), columns=COMP_ORDER,
                      index=(members["직원ID"] + "_" + members["직급"]).values)

st.dataframe(
    gap_df.style.background_gradient(cmap="RdBu_r", vmin=-1.5, vmax=1.5, axis=None)
                .format("{:.1f}"),
    use_container_width=True,
)

# ============================================================
# 3. 구성원별 교육 추천 (콘텐츠 기반)
# ============================================================
st.subheader(f"3. 구성원별 교육 추천 (콘텐츠 기반 Top {top_n})")

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

course_mat = tags.values
course_names = list(tags.index)

rows = []
for i, emp_label in enumerate(gap_df.index):
    gpos = np.clip(gap_df.iloc[i].values.astype(float), 0, None)
    if gpos.sum() == 0:
        rows.append({"구성원": emp_label, "부족역량": "—",
                     **{f"{k}순위": "(부족 역량 없음)" if k == 1 else "" for k in range(1, top_n + 1)}})
        continue
    scores = [(course_names[c], cosine(gpos, course_mat[c]) * gpos.sum())
              for c in range(len(course_names))]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_n]
    short = pd.Series(gpos, index=COMP_ORDER).sort_values(ascending=False).head(2)
    rec = {"구성원": emp_label,
           "부족역량": ", ".join(f"{c}({v:.1f})" for c, v in short.items() if v > 0)}
    for k, (cn, sc) in enumerate(top, 1):
        rec[f"{k}순위"] = f"{cn} ({sc:.2f})"
    rows.append(rec)

reco_df = pd.DataFrame(rows)
st.dataframe(reco_df, use_container_width=True, hide_index=True)

st.caption("요구역량=LDA 토픽 비중 환산 · 갭=요구−보유(각 0~5 정규화) · "
           "추천=부족역량과 강의 태그의 코사인 유사도 × 부족 총량")
