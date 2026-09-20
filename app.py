import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Dict

# =========================
# 基本設定（手機優先）
# =========================
st.set_page_config(
    page_title="台股決策 Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自訂 CSS：讓手機上字更大、卡片更清楚
st.markdown("""
<style>
    /* 大數字 metric */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    /* 標題稍微加大 */
    h1, h2, h3 {
        font-size: 1.4rem !important;
    }
    /* 卡片內距加大，比較好點 */
    .stMetric {
        padding: 10px 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 工具函數：形態驗證
# =========================
def detect_w_bottom(df: pd.DataFrame, window: int = 30) -> Tuple[int, bool]:
    if len(df) < window:
        df_use = df
    else:
        df_use = df.tail(window).copy()

    if len(df_use) < 15:
        return 0, False

    lows = df_use["Low"].values
    highs = df_use["High"].values
    closes = df_use["Close"].values

    mid = len(lows) // 2
    min1_idx = int(np.argmin(lows[:mid]))
    min2_idx = int(mid + np.argmin(lows[mid:]))

    low1 = lows[min1_idx]
    low2 = lows[min2_idx]

    if min2_idx > min1_idx + 1:
        neck = float(np.max(highs[min1_idx:min2_idx]))
    else:
        neck = float(np.max(highs))

    w_bottom = (low2 >= low1 * 0.98) and (closes[-1] > neck)
    score = 5 if w_bottom else 0
    return score, w_bottom


def detect_triangle_breakout(df: pd.DataFrame, window: int = 40) -> Tuple[int, bool]:
    if len(df) < window:
        df_use = df
    else:
        df_use = df.tail(window).copy()

    if len(df_use) < 20:
        return 0, False

    highs = df_use["High"].values
    lows = df_use["Low"].values
    closes = df_use["Close"].values

    mid = len(highs) // 2
    h1 = float(np.max(highs[:mid]))
    h2 = float(np.max(highs[mid:]))
    l1 = float(np.min(lows[:mid]))
    l2 = float(np.min(lows[mid:]))

    triangle = (h2 < h1) and (l2 > l1)
    breakout = triangle and (closes[-1] > h1)

    score = 5 if breakout else 0
    return score, breakout


# =========================
# 工具函數：動量評分
# =========================
def calculate_bb_width(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> float:
    close = df["Close"]
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bb_width = (upper - lower) / sma
    return float(bb_width.iloc[-1])


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1])


def momentum_score(df: pd.DataFrame) -> float:
    bb_width = calculate_bb_width(df)
    atr = calculate_atr(df)
    price = float(df["Close"].iloc[-1])

    # 簡化映射（可依歷史資料調整係數）
    bb_score = min(5.0, max(0.0, bb_width * 80))
    atr_score = min(5.0, max(0.0, (atr / price) * 800))

    return min(10.0, max(0.0, bb_score + atr_score))


# =========================
# 工具函數：Pivot Point
# =========================
def calculate_pivot_points(prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
    P = (prev_high + prev_low + prev_close) / 3.0
    R1 = 2 * P - prev_low
    S1 = 2 * P - prev_high
    R2 = P + (prev_high - prev_low)
    S2 = P - (prev_high - prev_low)
    R3 = prev_high + 2 * (P - prev_low)
    S3 = prev_low - 2 * (prev_high - P)

    return {
        "P": P,
        "R1": R1,
        "S1": S1,
        "R2": R2,
        "S2": S2,
        "R3": R3,
        "S3": S3,
    }


def pivot_score(df: pd.DataFrame, pivot_dict: Dict[str, float]) -> int:
    close = float(df["Close"].iloc[-1])
    P = pivot_dict["P"]
    R1 = pivot_dict["R1"]
    R2 = pivot_dict["R2"]
    S1 = pivot_dict["S1"]

    score = 0
    if close > P:
        score += 1
    if close > R1:
        score += 2
    if close > R2:
        score += 2
    if close < S1:
        score = max(0, score - 2)

    return min(5, max(0, score))


# =========================
# 工具函數：市場結構（量價相關）
# =========================
def market_structure_score(df: pd.DataFrame, window: int = 20) -> Tuple[int, str]:
    if len(df) < window:
        df_use = df
    else:
        df_use = df.tail(window).copy()

    volume = df_use["Volume"].values
    price_change = df_use["Close"].diff().values

    volume = volume[1:]
    price_change = price_change[1:]

    if len(volume) < 5:
        return 3, "盤整"

    corr = float(np.corrcoef(volume, price_change)[0, 1])
    if np.isnan(corr):
        return 3, "盤整"

    if corr > 0.5:
        return 5, "趨勢"
    elif corr < -0.3:
        return 1, "異常"
    else:
        return 3, "盤整"


# =========================
# 主程式：Dashboard（手機版面）
# =========================

# ----- 標題 -----
st.title("📊 台股決策 Dashboard")

# ----- 側邊欄：股票與參數 -----
st.sidebar.header("設定")

stock_list = ["2330.TW", "2454.TW", "0050.TW"]
selected_stock = st.sidebar.selectbox("選擇股票", stock_list)

st.sidebar.subheader("分數加權")
w_pattern = st.sidebar.slider("形態權重", 0.0, 1.0, 0.30, 0.05)
w_momentum = st.sidebar.slider("動量權重", 0.0, 1.0, 0.25, 0.05)
w_pivot = st.sidebar.slider("樞軸權重", 0.0, 1.0, 0.20, 0.05)
w_structure = st.sidebar.slider("結構權重", 0.0, 1.0, 0.25, 0.05)

total_w = w_pattern + w_momentum + w_pivot + w_structure
if total_w == 0:
    total_w = 1.0
w_pattern /= total_w
w_momentum /= total_w
w_pivot /= total_w
w_structure /= total_w

st.sidebar.info("加權總和已自動正規化為 1.00")

# ----- 載入資料（此處用假資料，之後可改接 CSV / API） -----
@st.cache_data
def load_data(stock_id: str) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(120))
    df = pd.DataFrame({
        "Date": dates,
        "Open": close + np.random.randn(120) * 0.5,
        "High": close + np.abs(np.random.randn(120)),
        "Low": close - np.abs(np.random.randn(120)),
        "Close": close,
        "Volume": np.random.randint(1000, 10000, 120)
    })
    return df

df = load_data(selected_stock)

# ----- 計算四大模塊分數 -----
pattern_w_score, w_flag = detect_w_bottom(df)
pattern_tri_score, tri_flag = detect_triangle_breakout(df)
pattern_score = max(pattern_w_score, pattern_tri_score)  # 0–5

mom_score = momentum_score(df)  # 0–10

prev = df.iloc[-2]
pivot_dict = calculate_pivot_points(
    float(prev["High"]),
    float(prev["Low"]),
    float(prev["Close"])
)
piv_score = pivot_score(df, pivot_dict)  # 0–5

struct_score, struct_type = market_structure_score(df)  # 0–5, 文字

# ----- 加權總分 -----
mom_score_5 = mom_score / 2.0

total_score_5 = (
    w_pattern * pattern_score +
    w_momentum * mom_score_5 +
    w_pivot * piv_score +
    w_structure * struct_score
)
total_score_10 = total_score_5 * 2.0

# =========================
# 手機版面配置重點
# =========================
# 1. 最上方直接放「進場建議」+「總分」
# 2. 再用一行四格顯示四個模塊分數
# 3. 圖表高度控制在 350–400px
# 4. 詳細資料全部收進 expander

# ----- 進場建議（手機第一眼看到） -----
st.subheader("🚦 進場建議")

threshold_buy = 7.0
threshold_watch = 5.0

# 用三欄式：左邊總分、中間訊號、右邊空白（避免太擠）
col_total, col_signal, _ = st.columns([1, 2, 1])

with col_total:
    st.metric("🎯 加權總分", f"{total_score_10:.2f}/10")

with col_signal:
    if total_score_10 >= threshold_buy:
        st.success(f"✅ 進場訊號：總分 ≥ {threshold_buy}")
    elif total_score_10 >= threshold_watch:
        st.warning(f"⚠️ 觀察區間：{threshold_watch}–{threshold_buy}")
    else:
        st.info(f"⛔ 暫緩進場：總分 < {threshold_watch}")

st.caption("進場閾值可依個人風險偏好調整。")

# ----- 評分總覽（四個模塊） -----
st.subheader("🧮 評分總覽")

col1, col2, col3, col4 = st.columns(4)

col1.metric("形態", f"{pattern_score}/5")
col2.metric("動量", f"{mom_score:.1f}/10")
col3.metric("樞軸", f"{piv_score}/5")
col4.metric("結構", f"{struct_score}/5")

# 市場結構類型用一行小字
st.caption(f"市場結構：{struct_type}（量價相關性評估）")

# ----- 股價走勢 + 樞軸線（圖表高度控制） -----
st.subheader("📈 股價走勢 & 樞軸線")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="股價"
))

colors = {
    "P": "white",
    "R1": "red",
    "R2": "darkred",
    "R3": "orange",
    "S1": "green",
    "S2": "darkgreen",
    "S3": "blue"
}

for key, value in pivot_dict.items():
    fig.add_hline(
        y=value,
        line_dash="dash",
        line_color=colors[key],
        annotation_text=key,
        annotation_font_color=colors[key]
    )

fig.update_layout(
    height=380,  # 手機建議 350–400
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=20, b=10),
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig, use_container_width=True)

# ----- 詳細資料（全部收合，避免手機太長） -----
with st.expander("查看詳細資料與樞軸數值"):
    st.dataframe(df.tail(10))
    st.json(pivot_dict)

# ----- 簡易交易日誌（手機也可輸入） -----
st.subheader("📒 簡易交易日誌")

with st.form("trade_form"):
    c1, c2 = st.columns(2)
    with c1:
        trade_date = st.date_input("交易日期")
        trade_type = st.selectbox("交易類型", ["進場-多", "進場-空", "出場-多", "出場-空"])
    with c2:
        trade_price = st.number_input("交易價格", min_value=0.0, step=0.1)
        trade_qty = st.number_input("股數", min_value=0, step=100)

    trade_note = st.text_area("備註", height=50)
    submitted = st.form_submit_button("送出記錄")

    if submitted:
        st.success(
            f"已記錄：{trade_date} {trade_type} @ {trade_price} x {trade_qty} 股"
        )

# ----- 頁尾 -----
st.caption("此 Dashboard 供研究與個人操作參考，不構成投資建議。")