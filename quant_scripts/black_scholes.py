import numpy as np
import math
import time
from scipy import log, exp, sqrt, stats

class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_sim(self):
        # 2 columns since payoff is max(0,S-E).
        # S = stock_price, E = strike_price.
        option_data = np.zeros([self.iterations, 2])

        #1D array with same length as number ofiterations.
        rand = np.random.normal(0, 1, [1, self.iterations])

        #equation for stock price, S(t)
        stock_price = self.S0*np.exp(self.T*(self.rf - 0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)

        # max(0, S-E)
        option_data[:,1] = stock_price - self.E

        #calc average return for the Monte-Carlo method. amax() returns max(S-E,0)
        average = np.sum(np.amax(option_data, axis = 1))/float(self.iterations)

        #have to use the exp(-rT) discount factor
        return np.exp(-1.0*self.rf*self.T)*average

    def put_option_sim(self):
        # 2 columns since payoff is max(0,E-S).
        option_data = np.zeros([self.iterations, 2])

        # 1D array with same length as number ofiterations.
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for stock price, S(t)
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        # S - E
        option_data[:, 1] = self.E - stock_price

        # calc average return for the Monte-Carlo method. amax() returns max(0,E-S)
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # Use exp(-rT) to calc the present value of the future cash flow.
        return np.exp(-1.0 * self.rf * self.T) * average

def call_option(S,E,T,rf,sigma):
    #calc d1 and d2 parameters for exact soln of BSE
    d1 = (log(S/E) + (rf + sigma**2 / 2.0) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    # stats.norm.cdf is the cumulative density function of the normal dist
    return S * stats.norm.cdf(d1) - E * exp(-rf * T) * stats.norm.cdf(d2)

def put_option(S,E,T,rf,sigma):
    #calc d1 and d2 parameters for exact soln of BSE
    d1 = (log(S/E) + (rf + sigma**2 / 2.0) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    # stats.norm.cdf is the cumulative density function of the normal dist
    return -S * stats.norm.cdf(-d1) + E * exp(-rf * T) * stats.norm.cdf(-d2)

if __name__ == '__main__':

    S0 = 100              #underlying stock price at t = 0
    E = 100               #strike price
    T = 1                 #expiry
    rf = 0.05             #risk free rate
    sigma = 0.2           #volatility of underlying stock
    iterations = 1000000  #number of iterations for MC.

    model = OptionPricing(S0, E, T, rf, sigma, iterations)
    print('Call option price with MC simulation: ', model.call_option_sim())
    print('Call option price with analytical solution: ', call_option(S0,E,T,rf,sigma))
    print('Put option price with MC simulation: ', model.put_option_sim())
    print('Put option price with analytical solution: ', put_option(S0,E,T,rf,sigma))
