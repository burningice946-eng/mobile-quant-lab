import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Quant Lab Mobile Dashboard", page_icon="📈", layout="centered"
)

st.title("📈 Mobile Quant Trading Lab")
st.markdown("Interactive momentum strategy dashboard powered by Python & Streamlit.")

# Sidebar Controls for User Interaction
st.sidebar.header("Strategy Settings")
ticker = st.sidebar.selectbox(
    "Select Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "MSFT", "NVDA", "SPY"]
)
fast_ma = st.sidebar.slider("Fast Moving Average", 2, 10, 2)
slow_ma = st.sidebar.slider("Slow Moving Average", 5, 30, 15)
period = st.sidebar.selectbox(
    "Historical Period", ["30d", "60d", "90d", "6mo", "1y"]
)


# Fetch Data with Caching
@st.cache_data
def load_data(tk, per):
  data = yf.download(tk, period=per, progress=False)
  if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
  df = data[["Close"]].dropna().copy()
  df.reset_index(inplace=True)
  if isinstance(df["Close"], pd.DataFrame):
    df["Close"] = df["Close"].iloc[:, 0]
  return df


df = load_data(ticker, period)

# Calculate Indicators & Returns
df["Fast_MA"] = df["Close"].rolling(fast_ma).mean()
df["Slow_MA"] = df["Close"].rolling(slow_ma).mean()
df["Signal"] = 0
df.loc[df["Fast_MA"] > df["Slow_MA"], "Signal"] = 1

df["Market_Return"] = df["Close"].pct_change()
df["Strategy_Return"] = df["Market_Return"] * df["Signal"].shift(1)

strat_returns = df["Strategy_Return"].dropna()
market_returns = df["Market_Return"].dropna()

strat_cum = (
    ((1 + strat_returns).prod() - 1) * 100 if len(strat_returns) > 0 else 0
)
market_cum = (
    ((1 + market_returns).prod() - 1) * 100 if len(market_returns) > 0 else 0
)

annual_factor = 365 if "-USD" in ticker else 252
sharpe = (
    (strat_returns.mean() / strat_returns.std()) * np.sqrt(annual_factor)
    if strat_returns.std() != 0
    else 0
)

# Display Key Metrics Cards
col1, col2, col3 = st.columns(3)
col1.metric("Strategy Return", f"{strat_cum:.2f}%")
col2.metric("Buy & Hold", f"{market_cum:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

# Render Interactive Charts
st.subheader(f"Price & Moving Averages ({ticker})")
st.line_chart(df.set_index("Date")[["Close", "Fast_MA", "Slow_MA"]])

st.subheader("Cumulative Strategy Equity Curve")
cum_equity = (1 + strat_returns.fillna(0)).cumprod()
st.line_chart(cum_equity)
