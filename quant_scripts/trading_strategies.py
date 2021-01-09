import numpy as np
import pandas as pd
import requests as r
from scipy import stats
import math
import os
from input_files.secrets import IEX_CLOUD_API_TOKEN
from datetime import datetime

class ConnectionError(Exception):
    """Raise if r.get doesn't return a 200 response."""
    pass

def chunks(lst, n):
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def format_batch_symbols(stocks, batch_length):
    """Reformat batch stock symbols into a list of comma separated stings of length 'batch_length."""

    symbol_groups = list(chunks(stocks, batch_length))
    symbol_strings = []
    for i in range(0, len(symbol_groups)):
        symbol_strings.append(','.join(symbol_groups[i]))
    return symbol_strings

def batch_api_call(symbol_string, types):
    """Convert 'types' to comma separated string and perform a batch api call."""

    types = ','.join(map(str, types))
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types={types}&token={IEX_CLOUD_API_TOKEN}'
    data = r.get(batch_api_call_url)

    #raise error if url response is not 200.
    if data.status_code != 200:
        raise ConnectionError('Batch API URL doesn\'t return a 200 response. Check \'symbol_string\' and \'types\'')

    data = data.json()
    return data

def format_strategy_df(df, data, symbol_string, df_columns, IEX_column_dict):
    """
    Append required data from each API call to dataframe.
    :param df: Dataframe to append to.
    :param data: IEX data.
    :param symbol_string: Comma separated string of stock symbols.
    :param df_columns: Dataframe columns.
    :param row_data: Row of data.
    :param IEX_column_dict: Dictionary used to index 'data'.
    :return: Dataframe.
    """

    # skip latestPrice because it is created by default. Do not want to exclude from IEX_column_dict and cause confusion.
    if 'latestPrice' in IEX_column_dict['quote']:
        IEX_column_dict['quote'].remove('latestPrice')

    for symbol in symbol_string.split(','):
        row_data = []
        stock_price = data[symbol]['quote']['latestPrice']
        num_of_stocks_to_buy = 'N/A'
        row_data.extend([symbol, stock_price, num_of_stocks_to_buy])

        for key in IEX_column_dict.keys():
            for metric in IEX_column_dict[key]:

                # keep as none type and impute with mean later.
                try:
                    metric_value = round(data[symbol][key][metric],6)
                except TypeError:
                    metric_value = data[symbol][key][metric]

                row_data.append(metric_value)
        row = pd.Series(row_data, index=df_columns)
        df = df.append(row, ignore_index=True)
    return df

def impute_with_mean(df, columns):
    """Impute column data with mean of that column."""

    if isinstance(columns, str):
        df.loc[df[columns].isin([None]), [columns]] = df[columns].mean()
        df[columns].fillna(df[columns].mean(), inplace=True)
    elif isinstance(columns, list):
        for column in columns:
            df.loc[df[column].isin([None]), [column]] = df[column].mean()
            df[column].fillna(df[column].mean(), inplace=True)
    return df

def remove_glamour_stocks(df):
    """Remove glamour stocks for the value betting strategy."""

    df.sort_values('PE_ratio', inplace=True)
    df = df[df['PE_ratio'] > 0]
    return df

def calc_row_percentiles(df, score_metric):
    """
    Calc percentile each point in each column in scope.
    Take the mean of these to calculate given score metric.
    """

    #get return percentiles as a wy to rank each stock.
    columns = [col for col in df.columns if col not in ['ticker', 'stock_price', 'num_of_shares_to_buy']]
    percentile_columns = [f'{col}_percentile' for col in columns]
    metrics = dict(zip(columns, percentile_columns))
    for row in df.index:
        for metric in metrics.keys():
            df = impute_with_mean(df, metric)
            df.loc[row, metrics[metric]] = stats.percentileofscore(df[metric], df.loc[row, metric]) / 100
    df[score_metric] = df[percentile_columns].mean(axis = 1)
    return df

def top_n_stocks(df, n, score_metric):
    """Get the top n stocks given a scoring metric."""

    df.sort_values(score_metric, ascending=False, inplace=True)
    df = df[:n]
    df.reset_index(drop=True, inplace=True)
    return df

def number_of_shares(portfolio_size, df):
    """Calculate the number of shares to buy of each stock using equal weights method."""

    position_size = portfolio_size/len(df)
    df['num_of_shares_to_buy'] = df['stock_price'].apply(lambda x: math.floor(position_size / x))

def pick_stocks(stocks, strategy_dict, batch_length, df_columns, IEX_column_dict):
    """
    Pick stocks based on a given strategy and scoring metric.
    :param stocks: Stock list. Typically an index.
    :param strategy_dict: Dictionary containing strategy information.
    :param batch_length: Number of stocks in a batch API call.
    :param df_columns: Dataframe columns.
    :param IEX_column_dict: Dictionary used to index 'data'.
    :return: Organised stocks dataframe.
    """

    strategy = list(strategy_dict.keys())[0]
    score_metric = strategy_dict[strategy]
    symbol_strings = format_batch_symbols(stocks['Ticker'], batch_length)

    # perform a batch API call and append and format this data in a dataframe.
    df = pd.DataFrame(columns=df_columns)
    for symbol_string in symbol_strings:
        data = batch_api_call(symbol_string, IEX_column_dict.keys())
        df = format_strategy_df(df, data, symbol_string, df_columns, IEX_column_dict)

    if strategy == 'value':
        df = remove_glamour_stocks(df)
        df['EV/EBITDA'] = df['enterprise_value'] / df['EBITDA']
        df['EV/GP'] = df['enterprise_value'] / df['gross_profit']
        df.drop(['enterprise_value', 'EBITDA', 'gross_profit'], axis=1, inplace=True)

    # get return percentiles, get top n stocks and calc number of shares to buy
    df = calc_row_percentiles(df, score_metric)
    return df

def main():
    base_path = 'C:\\Users\\scott\\PycharmProjects\\quantitative_modelling'
    inputs_path = os.path.join(base_path, 'input_files')
    batch_length = 100

    #df_columns and IEX columns must be in same order.
    # MOMENTUM
    df_columns = ['ticker', 'stock_price', 'num_of_shares_to_buy', 'one_year_price_return', 'six_month_price_return',
              'three_month_price_return', 'one_month_price_return']

    IEX_column_dict = {'quote' : ['latestPrice'],
                    'stats' : ['year1ChangePercent', 'month6ChangePercent', 'month3ChangePercent', 'month1ChangePercent']}

    #VALUE
    # df_columns = ['ticker', 'stock_price', 'num_of_shares_to_buy', 'PE_ratio', 'enterprise_value', 'EBITDA', 'gross_profit', 'PB_ratio', 'PS_ratio']
    #
    # IEX_column_dict = {'quote' : ['latestPrice', 'peRatio'],
    #                 'advanced-stats' : ['enterpriseValue', 'EBITDA', 'grossProfit', 'priceToBook', 'priceToSales']}

    strategy_dict = {'momentum' : 'HQM_score'}
    #strategy_dict = {'value' : 'RV_score'}
    strategy = list(strategy_dict.keys())[0]
    score_metric = strategy_dict[strategy]
    todays_date = datetime.today().strftime('%d_%m_%Y')
    stocks = pd.read_csv(os.path.join(inputs_path, 'sp_500_stocks.csv'))

    #organise stocks wrt the strategy.
    df = pick_stocks(stocks, strategy_dict, batch_length, df_columns, IEX_column_dict)

    #pick the top n stocks and save.
    n = 50
    portfolio_size = 1000000
    df = top_n_stocks(df, n, score_metric)
    number_of_shares(portfolio_size, df)
    df.to_csv(os.path.join(base_path, 'suggested_trades', f'{todays_date}_{strategy}_stocks.csv'), index = False)

if __name__ == '__main__':
    main()