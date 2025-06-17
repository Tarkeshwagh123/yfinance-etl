import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date

def fetch_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        data = df['Close']
    else:
        data = df[['Close']]
        data.columns = [tickers[0]]  
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

def calculate_metrics(data):
    returns = data.pct_change().dropna()
    historical_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = historical_return / volatility
    ytd_return = (data.iloc[-1] / data.iloc[0]) - 1
    max_drawdown = (data / data.cummax()).min() - 1
    
    three_years = 252 * 3
    five_years = 252 * 5

    if len(data) >= three_years:
        return_3y = (data.iloc[-1] / data.iloc[-three_years]) - 1
    else:
        return_3y = np.nan

    if len(data) >= five_years:
        return_5y = (data.iloc[-1] / data.iloc[-five_years]) - 1
    else:
        return_5y = np.nan
        
    summary = pd.DataFrame({
        "Historical Return 1Y": historical_return,
        "3Y return": return_3y,
        "5Y return": return_5y,
        "YTD return": ytd_return,
        "Standard Deviation (Volatility)": volatility,
        "Sharpe ratio (Risk Adjusted Return)": sharpe_ratio,
        "Maximum drawdown": max_drawdown
    })
    return summary

def plot_graphs(data, summary, tickers):
    st.subheader("Line Chart: Price History")
    st.line_chart(data)

    st.subheader("Bar Chart: Key Metrics")
    bar_metrics = summary[["Historical Return 1Y", "Standard Deviation (Volatility)", "YTD return"]]
    st.bar_chart(bar_metrics)

    st.subheader("Scatter Plot: Risk vs Return (Volatility vs Return)")
    summary_reset = summary.reset_index().rename(columns={"index": "Ticker"})
    fig = px.scatter(
        summary_reset,
        x="Standard Deviation (Volatility)",
        y="Historical Return 1Y",
        text="Ticker",
        color="Ticker",
        title="Risk vs Return: Volatility vs Return",
        labels={
            "Standard Deviation (Volatility)": "Volatility (Risk)",
            "Historical Return 1Y": "Return (1Y)"
        }
    )
    st.plotly_chart(fig)

    st.subheader("Drawdown Chart: Rolling Drawdowns")
    # Calculate rolling drawdowns for each ticker
    window = 252  # 1 year rolling window
    drawdown_df = pd.DataFrame()
    for col in data.columns:
        roll_max = data[col].cummax()
        drawdown = (data[col] / roll_max) - 1
        drawdown_df[col] = drawdown

    fig2 = px.line(
        drawdown_df,
        labels={"value": "Drawdown", "index": "Date", "variable": "Ticker"},
        title="Rolling Drawdowns (Downside Risk)"
    )
    st.plotly_chart(fig2)

    
    st.subheader("Maximum Drawdown per Fund")
    max_dd = drawdown_df.min()
    st.bar_chart(max_dd)

def main():
    st.title("Stocks Comparison Matrix & Analytics Dashboard")
    tickers = st.text_input("Enter stock tickers (comma separated):", value="SPY, QQQ, VTI, VOO")
    start_date = st.date_input("Start date", value=date(2020, 1, 1))
    end_date = st.date_input("End date", value=date(2025, 6, 1))

    if st.button("Fetch & Compare"):
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        data = fetch_data(ticker_list, start_date, end_date)
        if not data.empty:
            st.write("Raw Price Data", data)
            summary = calculate_metrics(data)
            st.subheader("Comparison Matrix")
            st.dataframe(summary.style.format("{:.2%}"))
            selected = st.multiselect("Choose tickers to visualize", options=list(summary.index), default=list(summary.index))
            if selected:
                plot_graphs(data[selected], summary.loc[selected], selected)
            else:
                st.warning("Please select at least one ticker to visualize.")
        else:
            st.error("No data found for the given tickers.")

if __name__ == "__main__":
    main()