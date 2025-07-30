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
import plotly.figure_factory as ff
import requests
from textblob import TextBlob
import boto3
import json
from botocore.exceptions import ClientError
from yfinance import tickers
from pdf_rag_chatbot import run_pdf_rag_chatbot

def initialize_bedrock_client():
    """Initialize AWS Bedrock client"""
    try:
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name='us-east-1',  # Change to your preferred region
            aws_access_key_id=st.secrets["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws_secret_access_key"]
        )
        return bedrock_client
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

def generate_ai_explanation(bedrock_client, metric_name, ticker, metric_value, comparison_data=None):
    """Generate AI explanation for a specific metric using AWS Bedrock Llama"""
    
    prompts = {
        "Historical Return 1Y": f"The fund {ticker} had a {metric_value:.1%} return over the past year. Explain in simple terms what this means for an investor. Is this performance good or bad?",
        "Sharpe ratio (Risk Adjusted Return)": f"The fund {ticker} has a Sharpe ratio of {metric_value:.2f}. Explain what this number tells us about the fund's risk-adjusted performance.",
        "Maximum drawdown": f"The fund {ticker} had a maximum drawdown of {metric_value:.1%}. Explain what this means and why investors should care.",
        "Standard Deviation (Volatility)": f"The fund {ticker} has a volatility of {metric_value:.1%}. Explain what this tells us about the fund's risk level.",
        "Dividend Yield (%)": f"The fund {ticker} has a dividend yield of {metric_value:.2f}%. Explain what this means for income-focused investors.",
        "Expense Ratio (%)": f"The fund {ticker} has an expense ratio of {metric_value:.2f}%. Explain how this affects investor returns over time.",
        "Beta": f"The fund {ticker} has a beta of {metric_value:.2f}. Explain what this tells us about how the fund moves relative to the market.",
        "Alpha": f"The fund {ticker} has an alpha of {metric_value:.2%}. Explain what this indicates about the fund's performance versus its benchmark."
    }
    
    base_prompt = prompts.get(metric_name, f"Explain the {metric_name} of {metric_value} for fund {ticker}.")
    
    full_prompt = f"{base_prompt} Keep your explanation simple and practical. Avoid technical jargon. Answer in 2-3 plain sentences without any formatting or special characters."
    
    request_body = {
        "prompt": full_prompt,
        "max_gen_len": 150,
        "temperature": 0.2,
        "top_p": 0.8
    }
    
    try:
        response = bedrock_client.invoke_model(
            modelId="us.meta.llama4-maverick-17b-instruct-v1:0",
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        explanation = response_body.get('generation', 'Unable to generate explanation.')
        
        # Clean up any unwanted formatting
        explanation = explanation.replace('```', '').replace('#', '').replace('*', '')
        explanation = explanation.replace('<', '').replace('>', '').replace('[', '').replace(']', '')
        
        return explanation.strip()
        
    except ClientError as e:
        return f"Error generating explanation: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def generate_comparative_analysis(bedrock_client, summary_df, selected_tickers):
    """Generate comparative analysis for selected funds"""
    
    comparison_text = "Fund Performance Data:\n"
    for ticker in selected_tickers:
        if ticker in summary_df.index:
            return_val = float(summary_df.loc[ticker, 'Historical Return 1Y'])
            volatility_val = float(summary_df.loc[ticker, 'Standard Deviation (Volatility)'])
            sharpe_val = float(summary_df.loc[ticker, 'Sharpe ratio (Risk Adjusted Return)'])
            expense_val = float(summary_df.loc[ticker, 'Expense Ratio (%)'])
            
            comparison_text += f"\n{ticker}:\n"
            comparison_text += f"- Annual Return: {return_val:.1%}\n"
            comparison_text += f"- Volatility: {volatility_val:.1%}\n"
            comparison_text += f"- Sharpe Ratio: {sharpe_val:.2f}\n"
            comparison_text += f"- Expense Ratio: {expense_val:.2f}%\n"
    
    prompt = f"""You are a financial advisor. Analyze these funds and provide clear investment insights.

{comparison_text}

Answer these questions in plain text in 3-4 sentences (no formatting, no HTML, no markdown. Do not display the questions to the user, as they need to see a comparative analysis of the funds.):

1. Which fund has the best risk-adjusted returns (highest Sharpe ratio)?
2. Which fund would you recommend for a conservative investor seeking lower volatility?
3. Which fund offers the best value in terms of low fees?
4. What is your overall recommendation?

Keep it concise and conversational. Do not repeat information or add unnecessary details.
Provide a straightforward analysis that any investor can understand. Use simple sentences and avoid technical jargon. Use Paragraph breaks for better readability.
Make sure the sentences are concise and to the point, without any unnecessary complexity or formatting. 
Avoid any redundant information, for example "The funds are all exchange-traded funds (ETFs) that track various stock market indices. QQQ tracks the Nasdaq-100 Index, SPY, VOO, and ^GSPC track the S&P 500 Index, and VTI tracks the CRSP US Total Market Index."
"""
    
    request_body = {
        "prompt": prompt,
        "max_gen_len": 200,
        "temperature": 0.1,
        "top_p": 0.7
    }
    
    try:
        response = bedrock_client.invoke_model(
            modelId="us.meta.llama4-maverick-17b-instruct-v1:0",
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        analysis = response_body.get('generation', 'Unable to generate comparative analysis.')
        
        # Clean up the response to remove any unwanted formatting
        analysis = analysis.replace('```', '').replace('#', '').replace('*', '')
        analysis = analysis.replace('<', '').replace('>', '')
        
        return analysis.strip()
        
    except Exception as e:
        return f"Error generating comparative analysis: {str(e)}"

def display_ai_insights(bedrock_client, summary, selected_tickers):
    """Display AI-generated insights for the comparison matrix"""
    
    st.subheader("🤖 AI-Generated Insights")
    
    insight_tabs = st.tabs(["📊 Metric Explanations", "🔍 Comparative Analysis", "💡 Investment Recommendations", "📰 News/Sentiment"])
    
    with insight_tabs[0]:
        st.markdown("**Click on any metric to get AI explanation:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_ticker = st.selectbox("Select Fund:", selected_tickers)
            
        with col2:
            metrics_list = [
                "Historical Return 1Y", "Sharpe ratio (Risk Adjusted Return)", 
                "Maximum drawdown", "Standard Deviation (Volatility)",
                "Dividend Yield (%)", "Expense Ratio (%)", "Beta", "Alpha"
            ]
            selected_metric = st.selectbox("Select Metric:", metrics_list)
        
        if st.button("Generate Explanation"):
            if selected_ticker in summary.index:
                metric_value = summary.loc[selected_ticker, selected_metric]
                
                with st.spinner("Generating AI explanation..."):
                    explanation = generate_ai_explanation(
                        bedrock_client, selected_metric, selected_ticker, metric_value
                    )
                
                st.info(f"**{selected_metric} for {selected_ticker}:**\n\n{explanation}")
    
    with insight_tabs[1]:
        st.markdown("**Comparative Analysis of Selected Funds:**")
        
        with st.spinner("Analyzing fund performance..."):
            analysis = generate_comparative_analysis(bedrock_client, summary, selected_tickers)
            
        st.write(analysis)
    
    with insight_tabs[2]:
        st.markdown("**Investment Recommendations:**")
        
        risk_tolerance = st.selectbox(
            "What's your risk tolerance?",
            ["Conservative", "Moderate", "Aggressive"]
        )
        
        investment_goal = st.selectbox(
            "What's your primary investment goal?",
            ["Income Generation", "Capital Growth", "Balanced Growth", "Retirement Planning"]
        )
        
        if st.button("Get AI Recommendation"):
            # Generate personalized recommendation based on user profile
            fund_summary = "Fund Performance Summary:\n"
            for ticker in selected_tickers:
                if ticker in summary.index:
                    return_val = float(summary.loc[ticker, 'Historical Return 1Y'])
                    volatility_val = float(summary.loc[ticker, 'Standard Deviation (Volatility)'])
                    sharpe_val = float(summary.loc[ticker, 'Sharpe ratio (Risk Adjusted Return)'])
                    expense_val = float(summary.loc[ticker, 'Expense Ratio (%)'])
                    dividend_val = float(summary.loc[ticker, 'Dividend Yield (%)'])
                    
                    fund_summary += f"\n{ticker}:\n"
                    fund_summary += f"- Annual Return: {return_val:.1%}\n"
                    fund_summary += f"- Volatility: {volatility_val:.1%}\n"
                    fund_summary += f"- Sharpe Ratio: {sharpe_val:.2f}\n"
                    fund_summary += f"- Expense Ratio: {expense_val:.2f}%\n"
                    fund_summary += f"- Dividend Yield: {dividend_val:.2f}%\n"

            # recommendation_prompt = f"""
            # You are a friendly financial advisor talking directly to a client. Based on the fund data and investor profile below, give a conversational recommendation.

            # {fund_summary}

            # Client Profile:
            # - Risk Tolerance: {risk_tolerance}
            # - Investment Goal: {investment_goal}

            # Write a direct recommendation as if you're speaking to the client. Start with something like "Based on your conservative approach and focus on income generation, I recommend..." 

            # Do not use bullet points, numbered lists, code blocks, or step-by-step analysis. Write in complete paragraphs as natural conversation. Do not include any technical formatting, code syntax, function definitions, or analytical frameworks. Just give a straightforward, friendly recommendation in 2-3 sentences about which fund(s) to choose and why.

            # """
            recommendation_prompt = f"""You are giving investment advice to a friend over coffee. They have {risk_tolerance.lower()} risk tolerance and want {investment_goal.lower()}.

            Here are the fund options:
            {fund_summary}

            Tell them which fund to pick and why in one simple paragraph. Start your response immediately with your recommendation. Do not write any code, functions, or technical formatting.
           Example: "I would recommend SPY because..."

            Your advice:"""

            
            request_body = {
                "prompt": recommendation_prompt,
                "max_gen_len": 300,
                "temperature": 0.2,
                "top_p": 0.8
            }
            
            try:
                with st.spinner("Generating personalized recommendation..."):
                    response = bedrock_client.invoke_model(
                        modelId="us.meta.llama4-maverick-17b-instruct-v1:0",
                        body=json.dumps(request_body),
                        contentType="application/json"
                    )
                    
                    response_body = json.loads(response['body'].read())
                    recommendation = response_body.get('generation', 'Unable to generate recommendation.')
                
                st.success("**Personalized Investment Recommendation:**")
                st.write(recommendation)
                
            except Exception as e:
                st.error(f"Error generating recommendation: {str(e)}")
    with insight_tabs[3]:
        st.subheader("Latest News & Sentiment")
        news_api_key = st.secrets["newsapi_key"] if "newsapi_key" in st.secrets else st.text_input("Enter your NewsAPI key:")
        if news_api_key:
            for ticker in selected_tickers:
                st.markdown(f"### {ticker} News")
                articles = fetch_news(ticker, news_api_key)
                if not articles:
                    st.write("No news found.")
                for article in articles:
                    sentiment = analyze_sentiment(article["title"] + " " + article["description"])
                    sentiment_label = "🟢 Positive" if sentiment > 0.1 else "🔴 Negative" if sentiment < -0.1 else "🟡 Neutral"
                    st.markdown(f"- [{article['title']}]({article['url']}) ({sentiment_label})")
                    st.caption(f"{article['publishedAt']}")
        else:
            st.info("Please provide a NewsAPI key to see news headlines.")

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

def fetch_news(ticker, api_key, max_articles=5):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={api_key}"
    response = requests.get(url)
    articles = []
    if response.status_code == 200:
        data = response.json()
        for article in data.get("articles", [])[:max_articles]:
            articles.append({
                "title": article["title"],
                "url": article["url"],
                "description": article["description"] or "",
                "publishedAt": article["publishedAt"]
            })
    return articles

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # -1 (negative) to 1 (positive)


def search_stocks(query):
    """Return a list of (symbol, name) strings matching the query using Finnhub API."""
    if not query or len(query) < 1:
        return []
    api_key = st.secrets["finnhub_api_key"] if "finnhub_api_key" in st.secrets else st.text_input("Enter your Finnhub API key:")
    if not api_key:
        return []
    url = f"https://finnhub.io/api/v1/search?q={query}&token={api_key}"
    try:
        resp = requests.get(url, timeout=5)
        results = resp.json().get("result", [])
        return [f"{item['symbol']} - {item.get('description', item['symbol'])}" for item in results if 'symbol' in item]
    except Exception:
        return []

def plot_graphs(data, summary, selected, start_date, end_date, sector_data):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price", 
        "📊 Metrics", 
        "⚖️ Risk/Return", 
        "📉 Drawdown", 
        "🏢 Sectors", 
        "💹 Returns"
    ])

    with tab1:
        st.subheader("Price Chart: Choose Visualization Type")
        chart_type = st.radio(
            "Select chart type:",
            ["Line (with SMA/EMA & Trend)", "Candlestick", "Area"],
            horizontal=True,
            key="chart_type"
        )

        import plotly.graph_objects as go

        for ticker in selected:
            df = data[[ticker]].copy()
            df['SMA50'] = df[ticker].rolling(window=50).mean()
            df['EMA20'] = df[ticker].ewm(span=20, adjust=False).mean()
            df['Uptrend'] = df[ticker] > df[ticker].shift(1)
            df['TrendIcon'] = df['Uptrend'].apply(lambda x: "⬆️" if x else "⬇️")

            if chart_type == "Line (with SMA/EMA & Trend)":
                with st.spinner("Loading Line chart..."):
                    time.sleep(1)
                    fig = go.Figure()
                    # Price line
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[ticker], mode='lines', name=f"{ticker} Price",
                        line=dict(color="#31333F", width=2)
                    ))
                    # SMA
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['SMA50'], mode='lines', name="SMA 50",
                        line=dict(color="#F28C3A", width=2, dash='dash')
                    ))
                    # EMA
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['EMA20'], mode='lines', name="EMA 20",
                        line=dict(color="#2ca02c", width=2, dash='dot')
                    ))
                    # Highlight up/down trends with markers
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[ticker],
                        mode='markers',
                        marker=dict(
                            color=df['Uptrend'].map({True: "#2ecc40", False: "#ff4136"}),
                            size=8,
                            symbol=df['Uptrend'].map({True: "triangle-up", False: "triangle-down"})
                        ),
                        name="Trend",
                        text=df['TrendIcon'],
                        showlegend=False
                    ))
                    fig.update_layout(
                        title=f"{ticker} Price with SMA/EMA and Trend Highlight",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Candlestick":
                # For candlestick, need OHLC data
                with st.spinner("Loading candlestick chart..."):
                    time.sleep(1)
                    hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
                    if not hist.empty and all(col in hist.columns for col in ["Open", "High", "Low", "Close"]):
                        fig = go.Figure(data=[go.Candlestick(
                            x=hist.index,
                            open=hist["Open"],
                            high=hist["High"],
                            low=hist["Low"],
                            close=hist["Close"],
                            name=f"{ticker} Candlestick"
                        )])
                        fig.update_layout(
                            title=f"{ticker} Candlestick Chart",
                            xaxis_title="Date",
                            yaxis_title="Price"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No OHLC data available for {ticker}.")

            elif chart_type == "Area":
                with st.spinner("Loading candlestick chart..."):
                    time.sleep(1)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[ticker], mode='lines', name=f"{ticker} Price",
                        line=dict(color="#F28C3A", width=2),
                        fill='tozeroy'
                    ))
                    fig.update_layout(
                        title=f"{ticker} Area Chart",
                        xaxis_title="Date",
                        yaxis_title="Price"
                    )
                    st.plotly_chart(fig, use_container_width=True)

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
        st.subheader("Sector Allocation Heatmap")
        sector_df = pd.DataFrame(sector_data).fillna(0)
        if not sector_df.empty:
            fig = px.imshow(sector_df, color_continuous_scale='Oranges', aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sector allocation data is not available for the selected tickers.")

    with tab6:
        st.subheader("Cumulative Returns (Growth of $1)")
        _, price_data = fetch_etf_metadata(selected, start_date, end_date)
        show_cumulative_returns(price_data)
        

def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Add these custom styles for the sidebar
    st.markdown("""
    <style>
    /* Set sidebar width and remove scrollbar */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #f5f7f9, #e8edf2);
        border-right: 1px solid #dfe5eb;
        box-shadow: 2px 0px 5px rgba(0,0,0,0.05);
        min-width: 320px !important;
        width: 320px !important;
    }
    
    /* Hide sidebar scrollbar but maintain functionality */
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto;
        scrollbar-width: none; /* Firefox */
        -ms-overflow-style: none;  /* IE and Edge */
    }
    
    /* Hide scrollbar for Chrome, Safari and Opera */
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
        display: none;
    }
    
    /* Sidebar header styling */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #31333F;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #dfe5eb;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Sidebar text styling */
    [data-testid="stSidebar"] .stMarkdown p {
        color: #4a5568;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }
    
    /* Dropdown styling */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        border-radius: 6px;
        border: 1px solid #dfe5eb;
        background-color: white;
        transition: border 0.2s;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border: 1px solid #F28C3A;
    }
    
    /* Button styling */
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 6px;
        background-color: #F28C3A;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        width: 100%;
        transition: background-color 0.2s;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #e07c2a;
    }
    
    /* Numbered list styling in sidebar */
    [data-testid="stSidebar"] ol {
        padding-left: 1.2rem;
        color: #4a5568;
    }
    
    [data-testid="stSidebar"] li {
        margin-bottom: 0.5rem;
    }
    
    /* Add some padding to sidebar content */
    [data-testid="stSidebarContent"] {
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    bedrock_client = initialize_bedrock_client()
    if 'fetch_compare' not in st.session_state:
        st.session_state['fetch_compare'] = False
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
    run_pdf_rag_chatbot(mode='button')
    #st.subheader("Input Parameters")
    tickers_input = st.text_input("Enter stock tickers (comma separated):", value="SPY, QQQ, VTI, VOO")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = st.date_input("Start date", value=date(2020, 1, 1))
    end_date = st.date_input("End date", value=date(2025, 6, 1))
    
    st.markdown("""
    <style>
    /* make each column a flex container centered on its items */
    div.stColumns > div {
        display: flex !important;
        align-items: center !important;
        margin-top: -30px !important;
    }
    /* shrink the horizontal gap between those two columns */
    div.stColumns {
        gap: 0.5rem !important;
        margin-top: -30px !important;
    }
    /* remove default button margins and normalize height */
    div.stButton > button {
        margin-top: -30px !important;
        height: 2.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # show Fetch & Compare and chat icon side by side
    # col1, col2 = st.columns([5, 9], gap="small")
    # with col1:
    #     if st.button("Fetch & Compare"):
    #         st.session_state['fetch_compare'] = True
    # with col2:
    #     run_pdf_rag_chatbot()
    col1, col2 = st.columns([5, 9], gap="small")
    with col1:
        if st.button("Fetch & Compare"):
            st.session_state['fetch_compare'] = True
    with col2:
        run_pdf_rag_chatbot(mode='popover')
    if st.session_state.get('fetch_compare', False):
        with st.spinner("Loading data..."):
            ticker_list = tickers
            data = fetch_data(tickers, start_date, end_date, benchmark=benchmark)
            if not data.empty:
                summary = calculate_metrics(data, tickers, start_date, end_date, benchmark=benchmark)
                all_tickers = list(summary.index)
                if bedrock_client:
                    display_ai_insights(bedrock_client, summary, all_tickers)
                    st.markdown("---")
                else:
                    st.warning("AWS Bedrock client not available. Please check your AWS credentials in Streamlit secrets.")

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
                            text-align: right;
                        }
                    </style>
                    """, unsafe_allow_html=True)
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
                    "Expense Ratio (%)":"{:.2f}"
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
                    sector_data, _ = fetch_etf_metadata(selected, start_date, end_date)
                    plot_graphs(data, summary, selected, start_date, end_date, sector_data)
                else:
                    st.warning("Please select at least one ticker to visualize.")
            else:
                st.error("No data found for the provided tickers and date range. Please check your input.")                                                                                                      

if __name__ == "__main__":
    main()