import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from interface.yfinance_client import YFinanceClient
from jobs.yfinance_etl.yfinances import transform_stock_data, load_stock_data

def extract_stock_data(ticker, start=None, end=None):
    client = YFinanceClient()
    data = client.fetch_stock_data(ticker, start=start, end=end)
    if data is None:
        print(f"No data extracted for {ticker} from {start} to {end}.")
        return None
    print(f"Extracted data for {ticker} from {start} to {end}:")
    print(data.head())
    return data

def etl_process(ticker, start=None, end=None):
    print(f"Starting ETL process for {ticker} from {start} to {end}")
    if not start:
        start = datetime.now().strftime('%Y-%m-%d')
    if not end:
        end = datetime.now().strftime('%Y-%m-%d')
    raw_data = extract_stock_data(ticker, start=start, end=end)
    if raw_data is None:
        return None
    transformed_data = transform_stock_data(raw_data)
    loaded_data = load_stock_data(transformed_data)
    return loaded_data

def main():
    tickers = ["AAPL", "GOOGL"]
    start_date = "2025-01-01"
    end_date = "2025-06-01"
    print(f"Running ETL for tickers1: {tickers} from {start_date} to {end_date}")
    print("Starting ETL process...")
    for ticker in tickers:
        try:
            result = etl_process(ticker, start=start_date, end=end_date)
            print(result)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()