import math as ma
import numpy as np
from numba import jit


@jit(nopython=True)
def binomialTree (r, sigma, S0, T, K, M, call = True, European = True):
    """Price a vanilla option using the Cox, Ross and Rubinstein binomial tree method, based
    on the algorithm presented in the Computational Finance script at the University of Kiel.

    Keyword arguments:
    r -- risk free rate
    sigma -- annual volatility
    S0 -- initial stock price
    T -- maturity of the option
    K -- strike price of the option
    call -- boolean value to differentiate between call and put options (default: call)
    European -- boolean value to differentiate between European and American options (default: European)
    M -- number of periods
    """

    # step 1: calibrate the lattice and the risk neutral probability
    dt = T/M
    alpha = ma.exp(r*dt)
    beta = (ma.pow(alpha,-1)+alpha*ma.exp((sigma**2)*dt))/2
    u = beta + ma.sqrt(beta**2-1)
    d = ma.pow(u,-1)
    q = (ma.exp(r*dt)-d)/(u-d)


    # step 2: build two empty lattices & initialize the first entry of the stock prices
    stockPrice = np.empty((M,M))
    stockPrice[0,0] = S0

    optionPrice = np.empty((M,M))

    
    # step 3: fill the lattice up with all possible stock prize realizations
    for i in range(M+1):
        for j in range(i+1):
            stockPrice[j,i] = S0 * u**j * d**(i-j)

    
    # step 4: compute the option value for every row in the last column
    if call == True:
        for j in range(M+1):
            optionPrice[j,M] = max(0, stockPrice[j,M]-K)

    else:
        for j in range(M+1):
            optionPrice[j,M] = max(0, K-stockPrice[j,M])
    

    # step 5: compute the option value backwardly
    if European == True:
        for i in reversed(range(M)):
            optionPrice[j,i] = ma.exp(-r*dt)*(q*optionPrice[j+1,i+1]+(1-q)*optionPrice[j,i+1])

    else:
        if call == True:
            optionPrice[j,i] = max(max(0,stockPrice[j,i]-K), ma.exp(-r*dt)*(q*optionPrice[j+1,i+1]+(1-q)*optionPrice[j,i+1]))
        else:
            max(max(0,K-stockPrice[j,i]), ma.exp(-r*dt)*(q*optionPrice[j+1,i+1]+(1-q)*optionPrice[j,i+1]))


    # step 6: return the option value at time t=0
    return optionPrice[0,0]




    



