#Fanghui Saho
#(a)
from scipy.stats import norm
import pandas as pd
import numpy as np
import copy
from scipy.optimize import root
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
F0 = [117.45, 120.60, 133.03, 152.05, 171.85]
S = [1, 2, 3, 4, 5]
def forward(F0):
    return 2 * np.log(0.5 * F0 / 10000 + 1)
f = list(map(forward, F0))

#(b)
def annuity(S, f):
    return sum(0.5 * np.exp(-0.5 * (S + i) * f) for i in range(1, 11)) 
annuities = list(map(annuity, S, f))

#(c)
def Bachelier(annu, sigma, T, F0_K):
    d1 = (F0_K) / (sigma * (T ** 0.5))
    return annu * sigma * (T ** 0.5) * (d1 * norm.cdf(d1) + norm.pdf(d1))

from_atm = [50, 25, 5, -5, -25, -50]

Fs = [[F0[i]]*6 for i in range(5)]

sigmas = {1:[58.31, 51.51, 49.28, 48.74, 41.46, 37.33], \
          2:[51.72, 46.87, 43.09, 42.63, 38.23, 34.55], \
          3:[46.29, 44.48, 43.61, 39.36, 35.95, 32.55], \
          4:[45.72, 41.80, 38.92, 38.19, 34.41, 31.15], \
          5:[44.92, 40.61, 37.69, 36.94, 33.36, 30.21]}

def premium(annu, sigma, F0_K):
    return [Bachelier(annu, sigma[i]/10000, 5, F0_K[i]/10000) for i in range(6)]

premiums = [premium(annuities[i], sigmas[i+1], from_atm) for i in range(5)]
premium_df = pd.DataFrame(premiums, columns = ['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])

Ks = copy.deepcopy(Fs)
for i in range(5):
    for j in range(6):
        Ks[i][j] -= from_atm[j]

# Q1(d)
def Approximation(T, K, F0, sigma0, alpha, beta, rho):
    Fmid = 0.5 * (F0 + K)
    C_mid = Fmid ** beta
    epsilon = T * alpha ** 2
    zeta = alpha / ( sigma0 * (1 - beta) ) * ( F0 ** (1 - beta) - K ** (1 - beta) )
    delta = np.log(  (np.sqrt(1 - 2 * rho * zeta + zeta ** 2) + zeta - rho) / (1 - rho) )
    gamma1 = beta / Fmid 
    gamma2 = beta * (beta - 1) / Fmid ** 2
    return alpha * (F0 - K) / delta * (1 + epsilon * ( (2 * gamma2 - gamma1 ** 2)/24) * (sigma0 * C_mid / alpha) ** 2 + rho * gamma1 * sigma0 * C_mid / 4 / alpha + (2 - 3 * rho ** 2)/24  )

def close_sigma(parameters, T, K_list, F0, sigma_list):
    beta = 0.5
    sigma0, alpha, rho = parameters
    return sum((Approximation(T, K_list[i]/10000, F0, sigma0, alpha, beta, rho) - sigma_list[i]/10000) ** 2 for i in range(6) )

sigma0_alpha_rho = [minimize(close_sigma, [0.1, 0.1, -0.1], args = (5, Ks[i], F0[i]/10000, sigmas[i+1]), method = 'SLSQP', bounds = ( (0.01, 1.5), (0, 1.5), (-1, 1) ) ).x for i in range(5)]

df = pd.DataFrame(sigma0_alpha_rho, columns = ['sigma0', 'alpha', 'rho'])
df['beta'] = 0.5

# Q1(f)

K_new = [[F0[i] - 75, F0[i] + 75 ] for i in range(5)]
Knew_df = pd.DataFrame(K_new, columns = ['ATM-75', 'ATM+75'])
vol = np.zeros((5, 2))
for i in range(5):
    for j in range(2):
        vol[i][j] = Approximation(5, K_new[i][j]/10000, F0[i]/10000, df['sigma0'][i], df['alpha'][i], df['beta'][i], df['rho'][i])
vol_df = pd.DataFrame(vol, columns = ['ATM-75', 'ATM+75'])

price = np.zeros((5, 2))
for i in range(5):
    for j in range(2):
        price[i][j] = Bachelier(annuities[i], vol[i][j], 5, [0.0075, -0.0075][j])
price_df = pd.DataFrame(price, columns = ['ATM-75', 'ATM+75'])

# Q1(g)
def BSM(A, sigma, T, F0, K):
    d1 = (np.log(F0 / K) + 0.5 * sigma ** 2 * T) / (sigma * T ** 0.5)
    d2 = (np.log(F0 / K) - 0.5 * sigma ** 2 * T) / (sigma * T ** 0.5)
    return A * (F0 * norm.cdf(d1) - K * norm.cdf(d2))
    
BsVol = np.zeros((5, 6))
for i in range(5):
    for j in range(6):
        BsVol[i][j] = root(lambda x: (BSM(annuities[i], x, 5, F0[i]/10000, Ks[i][j]/10000)[0] - premiums[i][j]), 0.1).x

BSVol_df = pd.DataFrame(BsVol, columns = ['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])
# Q1(h)

def BS_delta(annuity, sigma, T, F0, K):
    return annuity * norm.cdf((np.log(F0 / K) + 0.5 * sigma ** 2 * T) / (sigma * T ** 0.5))

deltas = np.zeros((5, 6))
for i in range(5):
    for j in range(6):
        deltas[i][j] = BS_delta(annuities[i], BsVol[i][j], 5, F0[i], Ks[i][j])

delta_df = pd.DataFrame(deltas, columns = ['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])
# Q1(i)

SABR_delta = np.zeros((5,6))
for i in range(5):
    F_up = F0[i]/10000 + 0.0001
    F_down = F0[i]/10000 - 0.0001
    for j in range(6):
        sigma_up = Approximation(5, Ks[i][j]/10000, F_up, df['sigma0'][i], df['alpha'][i], df['beta'][i], df['rho'][i])
        sigma_down = Approximation(5, Ks[i][j]/10000, F_down, df['sigma0'][i], df['alpha'][i], df['beta'][i], df['rho'][i])
        price_up = Bachelier(annuities[i], sigma_up, 5, F_up - Ks[i][j]/10000)
        price_down = Bachelier(annuities[i], sigma_down ,5, F_down - Ks[i][j]/10000)
        SABR_delta[i][j] = (price_up - price_down) / 0.0002

SABR_delta_df = pd.DataFrame(SABR_delta, columns = ['ATM-50', 'ATM-25', 'ATM-5', 'ATM+5', 'ATM+25', 'ATM+50'])

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 3, 1) 
plt.plot()
for i in range(1,6):
    plt.subplot(2, 3, i)
    plt.plot([-50,-25,-5,5,25,50],list(delta_df.iloc[0,:]),label='Black Model')
    plt.plot([-50,-25,-5,5,25,50],list(SABR_delta_df.iloc[0,:]),label='Smile Adjusted Delta')
    plt.title('Delta of swaption with Expiry = {} years'.format(i))
    plt.legend()
    
plt.show()





