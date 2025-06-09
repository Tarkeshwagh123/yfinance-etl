def format_stock_data(data):
    # Function to format stock data for display
    formatted_data = {
        "Ticker": data['symbol'],
        "Open": data['open'],
        "Close": data['close'],
        "High": data['high'],
        "Low": data['low'],
        "Volume": data['volume'],
        "Date": data['date'].strftime('%Y-%m-%d') if 'date' in data else None
    }
    return formatted_data

def validate_ticker(ticker):
    # Function to validate stock ticker input
    if not isinstance(ticker, str) or len(ticker) == 0:
        raise ValueError("Ticker must be a non-empty string.")
    return ticker.upper()