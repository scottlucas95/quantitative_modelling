import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as date
import modern_portfolio_theory as mp

risk_free_rate = 0.05

def capm(start_date, end_date, ticker1, ticker2):

    # get the data from yahoo finance
    stock1 = pdr.get_data_yahoo(ticker1, start=start_date, end=end_date)
    stock2 = pdr.get_data_yahoo(ticker2, start=start_date, end=end_date)

    # get monthly returns because long term investment (returns more likely normally distributed)
    return_stock1 = stock1.resample('M').last()
    return_stock2 = stock2.resample('M').last()

    #create dataframe from the data and get log returns
    data = pd.DataFrame({'s_adjclose' : return_stock1['Adj Close'], 'm_adjclose' : return_stock2['Adj Close']})
    data[['s_returns', 'm_returns']] = mp.daily_log_returns(data)
    data = data.dropna()

    # calc beta using the variance method
    # cov matrix and beta. Off diagonal is cov and [1,1] is variance of market
    covmat = data[['s_returns', 'm_returns']].cov().values
    beta_v = covmat[0,1]/covmat[1,1]

    # calc beta using the regression method
    beta_r, alpha = np.polyfit(data['m_returns'], data['m_returns'], deg = 1)

    #plot graph
    fig, axis = plt.subplots(1, figsize = (20,10))
    axis.scatter(data['m_returns'], data['s_returns'], label = 'Data Points')
    axis.plot(data['m_returns'], beta_r * data['m_returns'] + alpha, color = 'red', label = 'CAPM Line')
    plt.title('CAPM')
    plt.xlabel('Market Return')
    plt.ylabel('Stock returns')
    plt.text(0.08, 0.05, r'$R_a - \beta * R_m + \alpha$', fontsize = 10)
    plt.legend()
    plt.grid(True)
    plt.show()

    #*12 to get yearly mean return
    expected_return = risk_free_rate + beta_v * (data['m_returns'].mean()*12 - risk_free_rate)
    #expected return on stock = 9%  This shows that we can expect a better return if we take some more risk since 9 % > 5% (= risk free rate)
    print()


if __name__ == '__main__':
    capm('2010-01-01', '2017-01-01', 'IBM', '^GSPC')