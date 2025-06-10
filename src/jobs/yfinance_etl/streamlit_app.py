import streamlit as st
from yfinance_script import etl_process

def main():
    st.title("Compare ETL App")
    
    ticker = st.text_input("Enter Stock Ticker:", value="AAPL")
    start_date = "2025-01-01"
    end_date = "2025-06-01"
    if st.button("Fetch Data"):
        if ticker:
            data = etl_process(ticker,start=start_date, end=end_date)
            if data is not None:
                st.write(data)
            else:
                st.error("No data found for the given ticker.")
        else:
            st.error("Please enter a valid stock ticker.")

if __name__ == "__main__":
    main()