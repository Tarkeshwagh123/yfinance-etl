import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date

from yfinance import tickers

def fetch_data(tickers, start, end, benchmark="^GSPC"):
    all_tickers = tickers + [benchmark]
    df = yf.download(all_tickers, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        data = df['Close']
    else:
        data = df[['Close']]
        data.columns = [all_tickers[0]]  
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data


def fetch_etf_metadata(tickers, start_date, end_date):
    sector_data = {}
    holdings_data = {}
    price_data = {}

    for ticker in tickers:
        etf = yf.Ticker(ticker)
        # Sector Allocation
        sector_weights = etf.info.get('sectorWeightings', [])
        sectors = {list(s.keys())[0]: list(s.values())[0] * 100 for s in sector_weights} if sector_weights else {}
        sector_data[ticker] = sectors

        # Holdings (note: yfinance rarely returns this, so usually empty)
        top_holdings = etf.info.get('topHoldings', [])
        holdings_data[ticker] = top_holdings if top_holdings else "Not Available"

        # Price data for returns
        hist = etf.history(start=start_date, end=end_date)['Close']
        price_data[ticker] = hist

    return sector_data, holdings_data, price_data


def show_sector_allocation(sector_data):
    sector_df = pd.DataFrame(sector_data).fillna(0)
    st.subheader("Sector Allocation (%)")
    st.bar_chart(sector_df)

def show_cumulative_returns(price_data):
    returns_df = pd.DataFrame(price_data).pct_change().dropna()
    cumulative_returns = (1 + returns_df).cumprod()
    st.subheader("Cumulative Returns (Growth of $1)")
    st.line_chart(cumulative_returns)

def show_holdings(holdings_data):
    st.subheader("Top Holdings")
    for ticker, holdings in holdings_data.items():
        st.markdown(f"**{ticker} Top Holdings:**")
        if holdings == "Not Available":
            st.write("Not Available")
        else:
            st.write(pd.DataFrame(holdings))


def calculate_metrics(data, tickers, start_date, end_date, benchmark="^GSPC"):
    global dividends
    returns = data.pct_change().dropna()
    historical_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = historical_return / volatility
    ytd_return = (data.iloc[-1] / data.iloc[0]) - 1
    running_max = data.cummax()
    drawdown = (data - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Tracking Error calculation
    tracking_errors = {}
    if benchmark in data.columns:
        for ticker in data.columns:
            if ticker == benchmark:
                continue
            diff = returns[ticker] - returns[benchmark]
            daily_te = np.std(diff)
            annualized_te = daily_te * np.sqrt(252)
            tracking_errors[ticker] = annualized_te
    else:
        for ticker in data.columns:
            tracking_errors[ticker] = np.nan

    risk_free_rate = 0.01
    periods_per_year = 252
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.std(downside_returns)
    mean_excess_return = excess_returns.mean()
    sortino_ratio = mean_excess_return / downside_deviation

    dividend_yields = {}
    tickers = [ticker.strip().upper() for ticker in tickers.split(',') if ticker.strip()]
    # start_date = pd.to_datetime(start_date).tz_localize(None)
    # end_date = pd.to_datetime(end_date).tz_localize(None)
    for ticker in tickers:
        etf = yf.Ticker(ticker)
        print(etf.info.keys())
        # Get historical dividends
        dividends = etf.dividends
        dividends.index = dividends.index.tz_localize(None)
        dividends = dividends.loc[start_date:end_date]
        valid_dividends = dividends[dividends.index.isin(data.index)]
        valid_prices = data[ticker][data.index.isin(dividends.index)]

        # Calculate annual dividend and yield
        total_dividend = valid_dividends.sum()
        latest_price = data[ticker].iloc[-1]
        dividend_yield = (total_dividend / latest_price) * 100
        dividend_yields[ticker] = dividend_yield

    for ticker in tickers:
        fund = yf.Ticker(ticker)
        expense_ratio = fund.info.get("expenseRatio", 0)
        expense_ratio = expense_ratio * 100

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
        "Maximum drawdown": max_drawdown,
        "Sortino Ratio": sortino_ratio,
        "Dividend Yield (%)": dividend_yields,
        "Expense Ratio (%)": expense_ratio,
        "Tracking Error (%)": pd.Series(tracking_errors) * 100
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
    # max_dd = drawdown_df.min()
    st.bar_chart(drawdown_df)

    st.subheader("Dividend Yield (%)")
    yields = summary["Dividend Yield (%)"].sort_values(ascending=False)
    st.bar_chart(yields)

def main():
    benchmark = "^GSPC"
    st.title("Stocks Comparison Matrix & Analytics Dashboard")
    tickers = st.text_input("Enter stock tickers (comma separated):", value="SPY, QQQ, VTI, VOO")
    start_date = st.date_input("Start date", value=date(2020, 1, 1))
    end_date = st.date_input("End date", value=date(2025, 6, 1))

    if st.button("Fetch & Compare"):
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        data = fetch_data(ticker_list, start_date, end_date, benchmark=benchmark)
        if not data.empty:
            st.write("Raw Price Data", data)
            summary = calculate_metrics(data, tickers, start_date, end_date, benchmark=benchmark)
            st.subheader("Comparison Matrix")
            # st.dataframe(summary.style.format("{:.2%}"))
            format_dict = {
                "Historical Return 1Y": "{:.2%}",
                "3Y return": "{:.2%}",
                "5Y return": "{:.2%}",
                "YTD return": "{:.2%}",
                "Standard Deviation (Volatility)": "{:.2%}",
                "Sharpe ratio (Risk Adjusted Return)": "{:.2f}",
                "Sortino Ratio": "{:.2f}",
                "Maximum drawdown": "{:.2%}",
                "Dividend Yield (%)": "{:.2f}",
                "Expense Ratio (%)":"{:.2f}" # not percent format
            }
            st.dataframe(summary.style.format(format_dict))
            selected = st.multiselect("Choose tickers to visualize", options=list(summary.index), default=list(summary.index))
            if selected:
                plot_graphs(data[selected], summary.loc[selected], selected)
            else:
                st.warning("Please select at least one ticker to visualize.")
        else:
            st.error("No data found for the given tickers.")
            
        sector_data, holdings_data, price_data = fetch_etf_metadata(ticker_list, start_date, end_date)
        show_sector_allocation(sector_data)
        show_cumulative_returns(price_data)
        show_holdings(holdings_data)

if __name__ == "__main__":
    main()