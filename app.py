# -*- coding: utf-8 -*-
"""역량 진단 · 교육 추천 대시보드 (Streamlit) — 다크 네온 테마"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="역량 진단·교육 추천", page_icon="📊", layout="wide")

COMP_ORDER = ["사업관리", "고객관리", "품질생산관리", "AI/SW", "해외영업", "시험평가",
              "제어", "HR_AX", "열관리개발", "재무회계", "성능HW"]

st.markdown("""
<style>
.stApp { background:#0B0E14; color:#FAFAFA; }
.block-container { padding-top:2rem; }
h1,h2,h3,h4,h5,p,span,label,div { color:#FAFAFA; }
h1,h2,h3,h4 { letter-spacing:-0.5px; }
[data-testid="stSidebar"] { background:#11141C; border-right:1px solid #1E2230; }
[data-testid="stSidebar"] * { color:#FAFAFA !important; }
[data-testid="stSidebar"] small, [data-testid="stSidebar"] .stCaption { color:#9AA0AD !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background:#0B0E14 !important; border-color:#2A2E38 !important; color:#FAFAFA !important;
}
[data-baseweb="popover"] li { background:#11141C !important; color:#FAFAFA !important; }
.metric-card {
  background:linear-gradient(135deg,#141823 0%,#1B2130 100%);
  border:1px solid #232838; border-radius:16px; padding:18px 20px;
  box-shadow:0 4px 18px rgba(0,0,0,0.4);
}
.metric-card .label { color:#9AA0AD; font-size:13px; margin-bottom:6px; }
.metric-card .value { color:#2DE3A8; font-size:30px; font-weight:700; line-height:1.1; }
.metric-card .sub { color:#C9CDD6; font-size:12px; margin-top:4px; }
.section-tag {
  display:inline-block; background:linear-gradient(135deg,#2DE3A8,#1D9E75); color:#04342C;
  font-weight:700; font-size:12px; padding:3px 12px; border-radius:20px; margin-bottom:6px;
}
.subtle { color:#9AA0AD !important; }
[data-testid="stDataFrame"] { background:#141823; border-radius:12px; border:1px solid #232838; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    M = pd.read_excel("역량사전_LDA결과.xlsx", sheet_name="직무x역량_분포", index_col=0)
    M = M[[c for c in M.columns if c.startswith("역량후보")]]
    M.columns = COMP_ORDER
    job_base = (M * 5)
    emp = pd.read_csv("직원_역량프로파일_46직무.csv", encoding="utf-8-sig")
    tags = pd.read_csv("교육콘텐츠_역량태깅.csv", index_col=0)[COMP_ORDER]
    return job_base, emp, tags


def mm05(df):
    lo, hi = df.values.min(), df.values.max()
    return (df - lo) / (hi - lo) * 5 if hi > lo else df * 0


job_base, emp, tags = load_data()
요구_all = mm05(job_base).round(2)
보유_byjob = emp.groupby("직무")[[f"역량_{c}" for c in COMP_ORDER]].mean()
보유_byjob.columns = COMP_ORDER
보유_all = mm05(보유_byjob).round(2)
JOBS = list(요구_all.index)

st.sidebar.markdown("### ⚙️ 설정")
job = st.sidebar.selectbox("직무 선택", JOBS,
                           index=JOBS.index("HRM_채용인사관리") if "HRM_채용인사관리" in JOBS else 0)
top_n = st.sidebar.slider("1인당 추천 강의 수", 1, 5, 3)
st.sidebar.caption("LDA 토픽 기반 요구역량 · 가상 직원 데이터 · 콘텐츠 기반 추천")

st.markdown("# 📊 직무 역량 진단 · 교육 추천")
st.markdown(f'<p class="subtle">선택 직무 — <b style="color:#FAFAFA">{job}</b></p>', unsafe_allow_html=True)

members = emp[emp["직무"] == job].copy()
have_cols = [f"역량_{c}" for c in COMP_ORDER]
have_raw = members[have_cols].copy(); have_raw.columns = COMP_ORDER
have_05 = have_raw / have_raw.values.max() * 5 if have_raw.values.max() > 0 else have_raw
req_vec = 요구_all.loc[job]
gap = (req_vec.values[None, :] - have_05.values)
gap_df = pd.DataFrame(gap.round(2), columns=COMP_ORDER,
                      index=(members["직원ID"] + "_" + members["직급"]).values)

핵심역량 = req_vec.idxmax()
부족칸 = int((gap > 0.3).sum())
평균부족 = float(np.clip(gap, 0, None).mean())

c1, c2, c3, c4 = st.columns(4)
for col, label, value, sub in [
    (c1, "구성원 수", f"{len(members)}명", "선택 직무 인원"),
    (c2, "핵심 요구역량", 핵심역량, f"요구 {req_vec.max():.1f}/5"),
    (c3, "개발 필요 칸", f"{부족칸}개", "갭 0.3 초과"),
    (c4, "평균 부족도", f"{평균부족:.2f}", "0~5 척도"),
]:
    col.markdown(f"""<div class="metric-card"><div class="label">{label}</div>
    <div class="value">{value}</div><div class="sub">{sub}</div></div>""",
                 unsafe_allow_html=True)

st.write("")

# ---- 다크 차트 공통 설정 ----
def darkize(chart):
    return (chart.configure(background="#141823")
                 .configure_view(strokeWidth=0)
                 .configure_axis(labelColor="#C9CDD6", titleColor="#C9CDD6",
                                 gridColor="#222838", domainColor="#333B4D")
                 .configure_legend(labelColor="#C9CDD6", titleColor="#C9CDD6"))

st.markdown('<span class="section-tag">STEP 1</span>', unsafe_allow_html=True)
st.markdown("#### 직무 요구역량 vs 보유역량(평균)")
cmp_long = (pd.DataFrame({"요구역량": req_vec, "보유역량(평균)": 보유_all.loc[job]})
            .reset_index().melt(id_vars="index", var_name="구분", value_name="점수")
            .rename(columns={"index": "역량"}))
bar = (alt.Chart(cmp_long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
       .encode(
           x=alt.X("역량:N", sort=COMP_ORDER, axis=alt.Axis(labelAngle=-40)),
           xOffset="구분:N",
           y=alt.Y("점수:Q"),
           color=alt.Color("구분:N", scale=alt.Scale(
               domain=["요구역량", "보유역량(평균)"], range=["#2DE3A8", "#5B8DEF"]),
               legend=alt.Legend(orient="bottom")),
           tooltip=["역량", "구분", "점수"])
       .properties(height=340))
st.altair_chart(darkize(bar), use_container_width=True)

st.markdown('<span class="section-tag">STEP 2</span>', unsafe_allow_html=True)
st.markdown("#### 구성원별 역량 갭 (요구 − 보유) · 양수=부족")
def neon_diverging(v, vmax=1.5):
    # 부족(양수)=핫핑크/레드, 여유(음수)=시안, 0=거의 검정
    t = max(-1.0, min(1.0, v / vmax))
    if t >= 0:
        r = int(20 + t * 215); g = int(24 + t * 20); b = int(40 + t * 60)
    else:
        a = -t
        r = int(20 + a * 10); g = int(24 + a * 200); b = int(40 + a * 175)
    fg = "#FFFFFF" if abs(t) > 0.45 else "#C9CDD6"
    return f"background-color:rgb({r},{g},{b}); color:{fg}; font-weight:600;"

sty = (gap_df.style
       .map(lambda v: neon_diverging(float(v)))
       .format("{:.1f}")
       .set_table_styles([
           {"selector": "th", "props": [("background-color", "#11141C"),
                                        ("color", "#C9CDD6"), ("border-color", "#232838")]},
           {"selector": "td", "props": [("border-color", "#232838")]}]))
st.dataframe(sty, use_container_width=True)

st.markdown('<span class="section-tag">STEP 3</span>', unsafe_allow_html=True)
st.markdown(f"#### 구성원별 교육 추천 (콘텐츠 기반 Top {top_n})")

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

course_mat, course_names = tags.values, list(tags.index)
rows = []
for i, emp_label in enumerate(gap_df.index):
    gpos = np.clip(gap_df.iloc[i].values.astype(float), 0, None)
    if gpos.sum() == 0:
        rows.append({"구성원": emp_label, "부족역량": "—", "1순위": "(부족 역량 없음)"}); continue
    scores = sorted([(course_names[c], cosine(gpos, course_mat[c]) * gpos.sum())
                     for c in range(len(course_names))], key=lambda x: x[1], reverse=True)
    short = pd.Series(gpos, index=COMP_ORDER).sort_values(ascending=False).head(2)
    rec = {"구성원": emp_label,
           "부족역량": ", ".join(f"{c}({v:.1f})" for c, v in short.items() if v > 0)}
    for k, (cn, sc) in enumerate(scores[:top_n], 1):
        rec[f"{k}순위"] = f"{cn} ({sc:.2f})"
    rows.append(rec)

reco_df = pd.DataFrame(rows).fillna("")
reco_sty = (reco_df.style
            .set_properties(**{"background-color": "#141823", "color": "#FAFAFA"})
            .set_table_styles([
                {"selector": "th", "props": [("background-color", "#11141C"),
                                             ("color", "#2DE3A8"), ("font-weight", "700"),
                                             ("border-color", "#232838")]},
                {"selector": "td", "props": [("border-color", "#232838")]}]))
st.dataframe(reco_sty, use_container_width=True, hide_index=True)
st.caption("요구역량=LDA 토픽 비중 환산 · 갭=요구−보유(각 0~5 정규화) · 추천=부족역량과 강의 태그의 코사인 유사도 × 부족 총량")
