import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="台股決策 Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📊 台股交易決策 Dashboard")

st.sidebar.header("設定")
stock_list = ["2330.TW", "2454.TW", "0050.TW"]
selected_stock = st.sidebar.selectbox("選擇股票", stock_list)

# 假資料
dates = pd.date_range("2025-01-01", periods=100, freq="D")
np.random.seed(42)
close = 100 + np.cumsum(np.random.randn(100))
df = pd.DataFrame({
    "Date": dates,
    "Open": close + np.random.randn(100) * 0.5,
    "High": close + np.abs(np.random.randn(100)),
    "Low": close - np.abs(np.random.randn(100)),
    "Close": close,
    "Volume": np.random.randint(1000, 10000, 100)
})

st.subheader("📈 股價走勢")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="股價"
))
fig.update_layout(height=400, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.metric("🎯 總分", "7.2/10")
if 7.2 >= 7:
    st.success("✅ 進場訊號：總分 ≥ 7")
else:
    st.info("⛔ 暫緩進場：總分 < 7")