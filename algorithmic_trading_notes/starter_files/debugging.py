import pandas as pd
import numpy as np
import requests as r
from secrets import IEX_CLOUD_API_TOKEN
stocks = pd.read_csv('sp_500_stocks.csv')

my_columns = ['Ticker', 'Stock Price', 'Market Capitalisation', 'Number of Shares to Buy']
df = pd.DataFrame(columns = my_columns)
for stock in stocks['Ticker']:
    api_url = f'https://sandbox.iexapis.com/stable/stock/{stock}/quote/?token={IEX_CLOUD_API_TOKEN}'
    data = r.get(api_url).json()
    row = pd.Series([stock, data['latestPrice'], data['marketCap'], 'N/A'], index = my_columns)
    df = df.append(row, ignore_index = True)
print()