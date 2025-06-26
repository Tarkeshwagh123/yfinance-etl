import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date
from PIL import Image
import os
import statsmodels.api as sm
import time

from yfinance import tickers

@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def fetch_etf_metadata(tickers, start_date, end_date, retries=2):
    sector_data = {}
    price_data = {}

    for ticker in tickers:
        etf = yf.Ticker(ticker)
        # Retry logic for etf.info
        for attempt in range(retries):
            try:
                sector_weights = etf.info.get('sectorWeightings', [])
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    sector_weights = []
        sectors = {list(s.keys())[0]: list(s.values())[0] * 100 for s in sector_weights} if sector_weights else {}
        sector_data[ticker] = sectors

        # Price data for returns
        try:
            hist = etf.history(start=start_date, end=end_date)['Close']
            price_data[ticker] = hist
        except Exception:
            price_data[ticker] = pd.Series(dtype=float)

    return sector_data, price_data


def show_sector_allocation(sector_data):
    sector_df = pd.DataFrame(sector_data).fillna(0)
    if sector_df.empty or sector_df.sum().sum() == 0:
        st.info("Sector allocation data is not available for the selected tickers.")
    else:
        st.bar_chart(sector_df)

def show_cumulative_returns(price_data):
    returns_df = pd.DataFrame(price_data).pct_change().dropna()
    cumulative_returns = (1 + returns_df).cumprod()
    st.line_chart(cumulative_returns)

def show_holdings(holdings_data):
    st.subheader("Top Holdings")
    any_data = False
    for ticker, holdings in holdings_data.items():
        st.markdown(f"**{ticker} Top Holdings:**")
        if holdings == "Not Available" or holdings == []:
            st.write("Top holdings data is not available from Yahoo Finance for this ETF.")
        else:
            any_data = True
            st.write(pd.DataFrame(holdings))
    if not any_data:
        st.info("Top holdings data is not available for the selected tickers. This is a limitation of Yahoo Finance/yfinance.")


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
    ticker_objs = {ticker: yf.Ticker(ticker) for ticker in tickers}
    # start_date = pd.to_datetime(start_date).tz_localize(None)
    # end_date = pd.to_datetime(end_date).tz_localize(None)
    for ticker in tickers:
        etf = ticker_objs[ticker]
        #print(etf.info.keys())
        # Get historical dividends
        dividends = etf.dividends
        if isinstance(dividends.index, pd.DatetimeIndex):
            dividends.index = dividends.index.tz_localize(None)
            dividends = dividends.loc[start_date:end_date]
        else:
            dividends = pd.Series(dtype=float)  # Empty series if no dividends
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
		
	#alpha, beta, r^2
    benchmark_ticker = 'SPY'
    benchmark_data = fetch_data([benchmark_ticker], data.index[0], data.index[-1])
    benchmark_returns = benchmark_data.pct_change().dropna()

    returns = returns.loc[benchmark_returns.index.intersection(returns.index)]
    benchmark_returns = benchmark_returns.loc[returns.index]

    alphas = {}
    betas = {}
    r_squareds = {}

    for col in returns.columns:
        X = sm.add_constant(benchmark_returns.values)
        y = returns[col].values
        model = sm.OLS(y, X).fit()
        alphas[col] = model.params[0] * 252
        betas[col] = model.params[1]
        r_squareds[col] = model.rsquared


        
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
        "Tracking Error (%)": pd.Series(tracking_errors) * 100,
		"Alpha": pd.Series(alphas),
        "Beta": pd.Series(betas),
        "R-squared": pd.Series(r_squareds)
    })
    return summary

def plot_graphs(data, summary, selected, start_date, end_date):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Price History", 
        "Key Metrics Bar Chart", 
        "Risk vs Return Scatter", 
        "Drawdowns", 
        "Sector Allocation", 
        "Cumulative Returns"
    ])

    with tab1:
        st.subheader("Line Chart: Price History")
        st.line_chart(data[selected])

    with tab2:
        st.subheader("Bar Chart: Key Metrics")
        bar_metrics = summary.loc[selected, ["Historical Return 1Y", "Standard Deviation (Volatility)", "YTD return"]]
        st.bar_chart(bar_metrics)

    with tab3:
        st.subheader("Scatter Plot: Risk vs Return (Volatility vs Return)")
        summary_reset = summary.reset_index().rename(columns={"index": "Ticker"})
        fig = px.scatter(
            summary_reset.loc[summary_reset['Ticker'].isin(selected)],
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
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Drawdown Chart: Rolling Drawdowns")
        drawdown_df = pd.DataFrame()
        for col in data[selected].columns:
            roll_max = data[selected][col].cummax()
            drawdown = (data[selected][col] / roll_max) - 1
            drawdown_df[col] = drawdown
        fig2 = px.line(
            drawdown_df,
            labels={"value": "Drawdown", "index": "Date", "variable": "Ticker"},
            title="Rolling Drawdowns (Downside Risk)"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Maximum Drawdown per Fund")
        st.bar_chart(drawdown_df)

    with tab5:
        st.subheader("Sector Allocation (%)")
        sector_data, _ = fetch_etf_metadata(selected, start_date, end_date)
        show_sector_allocation(sector_data)

    with tab6:
        st.subheader("Cumulative Returns (Growth of $1)")
        _, price_data = fetch_etf_metadata(selected, start_date, end_date)
        show_cumulative_returns(price_data)
        

def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    load_custom_css()
    logo = Image.open("logo.png")
    st.image(logo, width=160)
    st.markdown(
        '<h1 style="font-size:2.0rem; color:#31333F; font-weight:bold; margin-bottom: 0.5em;">'
        'Stocks Comparison Matrix & Analytics Dashboard'
        '</h1>',
        unsafe_allow_html=True
    )
    
    benchmark = "^GSPC"
    st.sidebar.header("Settings")
    benchmark = st.sidebar.selectbox("Select Benchmark", options=["^GSPC", "^DJI", "^IXIC"], index=0)

    st.sidebar.subheader("Instructions")
    st.sidebar.markdown("""
    1. Enter stock tickers in the input box, separated by commas (e.g., SPY, QQQ, VTI).
    2. Select the start and end dates for the analysis.
    3. Click on 'Fetch & Compare' to retrieve data and generate the comparison matrix.
    4. Use the sidebar to select a benchmark for comparison.
    """)
    #st.sidebar.image("logo.png", width=180) 

    #st.subheader("Input Parameters")
    tickers_input = st.text_input("Enter stock tickers (comma separated):", value="SPY, QQQ, VTI, VOO")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = st.date_input("Start date", value=date(2020, 1, 1))
    end_date = st.date_input("End date", value=date(2025, 6, 1))

    if st.button("Fetch & Compare"):
        with st.spinner("Loading data..."):
            ticker_list = tickers
            data = fetch_data(tickers, start_date, end_date, benchmark=benchmark)
            if not data.empty:
                st.subheader("Raw Price Data")
                st.dataframe(
                    data.style.format("{:.2f}").set_properties(**{
                        'background-color': '#fff',
                        'color': '#F28C3A',
                        'border-color': '#F28C3A',
                        'font-size': '25px !important'
                    }),
                    use_container_width=True,
                    hide_index=False
                )
                summary = calculate_metrics(data, tickers, start_date, end_date, benchmark=benchmark)
                st.subheader("Comparison Matrix")
                st.markdown(
                    """
                    <style>
                        .stDataFrame {
                            font-size: 22px! important;
                            background-color: #fff;
                            color: #F28C3A;
                            border: 1px solid #F28C3A;
                            border-radius: 10px;
                        }
                    </style>
                    """, unsafe_allow_html=True)
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
                #st.subheader("Comparison Matrix")
                
                st.dataframe(
                    summary.T.style.format(format_dict).set_properties(**{
                        'background-color': '#fff',
                        'color': '#F28C3A',
                        'border-color': '#F28C3A',
                        'font-size': '25px !important'
                    }),
                    use_container_width=True,
                    hide_index=False,
                    height=summary.T.shape[0]*35 + 45
                )
                selected = st.multiselect("Choose tickers to visualize", options=list(summary.index), default=list(summary.index))
                if selected:
                    plot_graphs(data, summary, selected, start_date, end_date)
                else:
                    st.warning("Please select at least one ticker to visualize.")
            else:
                st.error("No data found for the provided tickers and date range. Please check your input.")                                                                                                      

if __name__ == "__main__":
    main()