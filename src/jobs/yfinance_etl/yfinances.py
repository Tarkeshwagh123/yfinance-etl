def transform_stock_data(data):
    # Transform the data into a suitable format
    print(data.columns)
    row  = data.iloc[0] 
    row_dict = row.to_dict()  
    if 'Date' in data.columns:
        date_value = row['Date']
    else:
        date_value = data.index[0]

    
    if hasattr(date_value, 'to_pydatetime'):
        date_value = date_value.to_pydatetime()

    row_dict['Date'] = date_value
    print(row_dict)
    # Transform the data into a more structured format
    row_dict['Date'] = date_value
    print(row_dict)
    
    transformed_data = {
        "Ticker": "",  
        "Open": row_dict.get('Open', ''),
        "Close": row_dict.get('Close', ''),
        "High": row_dict.get('High', ''),
        "Low": row_dict.get('Low', ''),
        "Volume": row_dict.get('Volume', ''),
        "Date": row_dict['Date'].strftime('%Y-%m-%d') if hasattr(row_dict['Date'], 'strftime') else str(row_dict['Date'])
    }
    return transformed_data

def load_stock_data(transformed_data):
    return transformed_data