# generate random weights using monte carlo and calc above step for each and plot
# find optimal porfolio
# find min sharpe ratio
# optimize portfolio using constraints (sum of weights = 1, minimise sharpe ratio)
# plot solution

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, optimize
import yfinance as yf
from pandas_datareader import data as pdr


def connect_to_stock_data(start_date, end_date, stocks):
    """Returns adjusted closing price of stocks between two dates."""

    yf.pdr_override()
    data = pdr.get_data_yahoo(stocks, start=start_date, end=end_date)['Adj Close']
    data.dropna(inplace=True)
    return data

def daily_log_returns(data):
    """Calculates the daily return of a stock and normalises using log."""

    daily_returns = np.log(data/data.shift(1))
    daily_returns.dropna(inplace = True)
    return daily_returns

def check_normally_distributed(data):
    #No matter what gets passed, we get p_value < O(-10) using statistical tests.
    #plotting the histogram shows normally distributed.

    for col in data.columns:
        daily_returns = daily_log_returns(data[col])
        #uniform= pd.Series(stats.uniform.rvs(size = 1000))
        #normal = pd.Series(stats.norm.rvs(size = 1000))

        #plot using .hist.
        #daily_returns.hist(bins=100)
        #uniform.hist(bins=100)
        #normal.hist(bins=100)

        #shapiro test of normality.  Most powerful normality test (apparently)
        #Can use stats.anderson() as an alternative test for normality.

        shapiro_daily_returns = stats.shapiro(daily_returns)
        #shapiro_uniform = stats.shapiro(uniform)
        #shapiro_normal = stats.shapiro(normal)
        print()

        # if p > alpha:  # alternative hypothesis: data doesn't comes from a normal distribution
        #     raise ValueError('{} returns are not normally distributed'.format(daily_returns.columns))

def show_statistics(returns):
    """
    Calc mean and cov for a given set of stocks over a trading year (252 days).
    - Mean is the expecting returns on a stock.
    - Diagonal entries is the variance for each stock.
    - Off diagonals is the covariance between pairs of stocks.
    """

    return returns.mean()*252, returns.cov()*252

def initialise_weights(stocks):
    """ Generates randoms weights and normalises them."""

    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    return weights

def calculate_portfolio_stats(returns, weights):
    """
    Calculates the expected return, variance and Sharpe ratio of the portfolio.
    :param returns: Stock returns.
    :param weights: Weights.
    :return: portfolio return, variance, Sharpe ratio.
    """

    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    portfolio_sharpe_ratio = portfolio_return / portfolio_variance
    return portfolio_return, portfolio_variance, portfolio_sharpe_ratio

def generate_portfolios(stocks, returns, iterations):
    """
    Use Monte Carlo simulation to simulate portfolio weights and output portfolio expected return, variance and Sharpe ratio.
    :param stocks: Stocks in the portfolio.
    :param returns: Portfolio returns.
    :return: Array of portfolio returns and variances.
    """

    n_portfolio_returns = []
    n_portfolio_variances = []

    for i in range(iterations):
        weights = initialise_weights(stocks)
        portfolio_return, portfolio_variance, portfolio_sharpe_ratio = calculate_portfolio_stats(returns, weights)
        n_portfolio_returns.append(portfolio_return)
        n_portfolio_variances.append(portfolio_variance)

    n_portfolio_returns = np.array(n_portfolio_returns)
    n_portfolio_variances = np.array(n_portfolio_variances)
    return n_portfolio_returns, n_portfolio_variances

def min_func_sharpe(weights, returns):
    """
    Optimise portfolio return wrt the Sharpe ratio.
    :param weights: Portfolio weights.
    :param returns: Daily returns.
    :return: -Sharpe ratio so we have a function to minimise.
    """

    portfolio_return, portfolio_variance, portfolio_sharpe_ratio = calculate_portfolio_stats(returns, weights)
    return -portfolio_sharpe_ratio

def optimum_portfolio_weights(stocks, returns):
    """
    Create weights, apply constraints and minimise objective function.
    :param stocks: Stocks in portfolio.
    :param returns: Daily returns - the argument to the minimum function.
    :return: Portfolio with optimum Sharpe ratio.
    """

    weights = initialise_weights(stocks)
    constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) - 1})     # make sure sum of weights is 1
    bounds = tuple((0,1) for c in range(len(stocks)))       #the weight of a given stock can be at most 1
    optimum = optimize.minimize(fun = min_func_sharpe, x0 = weights, args = returns, method = 'SLSQP', bounds = bounds, constraints = constraints)
    return optimum

def print_optimal_portfolio(optimum, returns):
    """Print optimal portfolio details."""

    portfolio_return, portfolio_variance, portfolio_sharpe_ratio = calculate_portfolio_stats(returns, optimum.x)
    print('Optimum weights: {}'.format(optimum.x.round(3)))
    print('Expected return, volatility and Sharpe ratio: {}, {}, {}'.format(portfolio_return, portfolio_variance, portfolio_sharpe_ratio))

def show_optimal_porfolio(optimum, returns, portfolio_returns, portfolio_variances, portfolio_sharpe_ratio):
    """Plot and label optimal portfolio."""

    plt.figure(figsize = (10,6))
    plt.scatter(portfolio_variances, portfolio_returns, c = portfolio_sharpe_ratio, marker = 'o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.plot(calculate_portfolio_stats(returns, optimum.x)[1], calculate_portfolio_stats(returns, optimum.x)[0], 'g*', markersize = 20)
    plt.show()

def main():
    # TODO: need to write a bit of code that, given all generated portfolios, gets the max return for a given risk and vice versa
    stocks = ['KO', 'ALB', 'SIVB', 'ALGN', 'MU']
    #stocks = ['APPL', 'KO', 'CE', 'A', 'BK']
    start = pd.to_datetime('2013-12-12')
    end = pd.to_datetime('2020-08-21')
    MC_iterations = 1000

    data = connect_to_stock_data(start, end, stocks)
    check_normally_distributed(data)
    daily_returns = daily_log_returns(data)
    n_portfolio_returns, n_portfolio_variances = generate_portfolios(stocks, daily_returns, MC_iterations)
    n_portfolio_sharpe_ratio = n_portfolio_returns / n_portfolio_variances
    optimum_weights = optimum_portfolio_weights(stocks, daily_returns)
    print_optimal_portfolio(optimum_weights, daily_returns)
    show_optimal_porfolio(optimum_weights, daily_returns, n_portfolio_returns, n_portfolio_variances, n_portfolio_sharpe_ratio)
    print()

if __name__ == '__main__':
    main()


