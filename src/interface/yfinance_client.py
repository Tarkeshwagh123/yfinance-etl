import yfinance as yf

class YFinanceClient:
    def fetch_stock_data(self, ticker):
        stock = yf.Ticker(ticker)
        data = stock.history(period="1mo")  # Fetching 1 month of historical data
        return data.reset_index()  # Reset index to have a clean DataFrame