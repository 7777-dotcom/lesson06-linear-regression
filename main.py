import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 해
MIN_DAYS = 300        # 한 해에 필요한 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly_mean():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])

    # 기준 기간(2025년까지)만 사용
    df = df[df["날짜"].dt.year <= LAST_YEAR].copy()
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    kept = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    dropped = yearly[yearly["관측일수"] < MIN_DAYS]["연도"].tolist()
    return kept, dropped


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온으로 회귀 직선을 만들고, 선택한 해의 기온을 예상해 봅니다.")

try:
    yearly, dropped_years = load_yearly_mean()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들 데이터가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

# 최근 20년(끝 연도 포함 20개 연도 구간)만으로 구한 회귀
RECENT_SPAN = 20
recent_start = end_year - RECENT_SPAN + 1
recent = yearly[yearly["연도"] >= recent_start]
has_recent = len(recent) >= 3
if has_recent:
    xr = recent["연도"].to_numpy(dtype=float)
    yr = recent["연평균기온"].to_numpy(dtype=float)
    slope_recent, intercept_recent = np.polyfit(xr, yr, 1)

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# ---- 슬라이더와 예상 기온 ----
year = st.slider("예상할 연도", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.2rem; opacity:0.75;">{year}년 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- 요약 지표 ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")


def slope_card(title, per_century, sub):
    return f"""
    <div style="text-align:center; padding:1rem; border:1px solid rgba(128,128,128,0.35);
                border-radius:12px;">
        <div style="font-size:1.1rem; opacity:0.75;">{title}</div>
        <div style="font-size:3.2rem; font-weight:800; line-height:1.25;">{per_century:+.2f} °C</div>
        <div style="font-size:1rem; opacity:0.75;">100년당 변화</div>
        <div style="font-size:0.85rem; opacity:0.6; margin-top:0.3rem;">{sub}</div>
    </div>
    """


st.subheader("100년에 몇 도 오르는가")
col_all, col_recent = st.columns(2)
col_all.markdown(
    slope_card(
        "전체 기간",
        slope * 100,
        f"{start_year}~{end_year}년, {n_years}개 해 · 기울기 {slope:+.4f} °C/년",
    ),
    unsafe_allow_html=True,
)
if has_recent:
    col_recent.markdown(
        slope_card(
            f"최근 {RECENT_SPAN}년",
            slope_recent * 100,
            f"{int(recent['연도'].min())}~{end_year}년, {len(recent)}개 해 · "
            f"기울기 {slope_recent:+.4f} °C/년",
        ),
        unsafe_allow_html=True,
    )
    diff = (slope_recent - slope) * 100
    ratio_txt = f"{slope_recent / slope:.1f}배" if slope > 0 else "비교 불가"
    st.write(
        f"최근 {RECENT_SPAN}년의 상승 속도는 전체 기간보다 100년당 **{diff:+.2f} °C** 차이가 나요 "
        f"(전체 기간의 {ratio_txt}). 해 수가 적은 구간이라 값이 크게 흔들릴 수 있어요."
    )
else:
    col_recent.info(f"최근 {RECENT_SPAN}년 구간에 남은 해가 3개 미만이라 계산할 수 없어요.")

st.write(f"전체 기간 회귀 직선: **기온 = {slope:.4f} × 연도 + ({intercept:.2f})**")

# ---- 그래프 ----
line_x0 = min(start_year, year)
line_x1 = max(end_year, year)
line_x = np.array([line_x0, line_x1], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, opacity=0.8),
        customdata=yearly["관측일수"],
        hovertemplate="%{x}년<br>%{y:.2f} °C<br>관측일 %{customdata}일<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, dash="dash"),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", line=dict(width=1, color="black")),
        hovertemplate="%{x}년 예상<br>%{y:.2f} °C<extra></extra>",
    )
)
if has_recent:
    rx = np.array([recent["연도"].min(), end_year], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=rx,
            y=slope_recent * rx + intercept_recent,
            mode="lines",
            name=f"최근 {RECENT_SPAN}년 직선",
            line=dict(width=3, dash="dot"),
            hoverinfo="skip",
        )
    )
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=10),
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"사용 기준: {LAST_YEAR}년까지의 자료, 관측일 {MIN_DAYS}일 이상인 해만 포함."
    + (f" 제외된 해: {', '.join(map(str, dropped_years))}" if dropped_years else "")
)
