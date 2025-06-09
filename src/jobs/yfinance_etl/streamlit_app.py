import streamlit as st
from yfinance_script import etl_process

def main():
    st.title("Compare ETL App")
    
    ticker = st.text_input("Enter Stock Ticker:", value="AAPL")
    
    if st.button("Fetch Data"):
        if ticker:
            data = etl_process(ticker)
            if data is not None:
                st.write(data)
            else:
                st.error("No data found for the given ticker.")
        else:
            st.error("Please enter a valid stock ticker.")

if __name__ == "__main__":
    main()