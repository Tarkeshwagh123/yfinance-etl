import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from interface.yfinance_client import YFinanceClient
from jobs.yfinance_etl.yfinances import transform_stock_data, load_stock_data

def extract_stock_data(ticker):
    #from interface.yfinance_client import YFinanceClient

    client = YFinanceClient()
    data = client.fetch_stock_data(ticker)
    return data

def etl_process(ticker):
    raw_data = extract_stock_data(ticker)
    transformed_data = transform_stock_data(raw_data)
    loaded_data = load_stock_data(transformed_data)
    return loaded_data

def main():
    tickers = ["AAPL", "GOOG"]
    for ticker in tickers:
        try:
            result = etl_process(ticker)
            print(result)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()