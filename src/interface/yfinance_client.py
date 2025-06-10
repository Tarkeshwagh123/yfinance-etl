import yfinance as yf

class YFinanceClient:
    # def fetch_stock_data(self, ticker, start=None, end=None):
    #     stock = yf.Ticker(ticker)
    #     data = stock.history(period=None, start=start, end=end)  
    #     print(f"Fetched data for {ticker} from {start} to {end}:")
    #     print(data.head())
    #     return data.reset_index()
    
    def fetch_stock_data(self, ticker, start=None, end=None, interval="1d"):
        print(f"Fetching data for {ticker} from {start} to {end} with interval {interval}")
        try:
            yf.enable_debug_mode()  # Enable debug mode for detailed logging
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False
            )
            if df.empty:
                print(f"No data found for {ticker} in the specified date range.")
                return None
            print(f"Data fetched for {ticker}:")
            print(df.head())
            return df.reset_index()
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None