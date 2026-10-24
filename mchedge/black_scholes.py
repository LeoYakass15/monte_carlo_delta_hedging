"""Black-Scholes formulas for a European call or put (no dividends).

They are the benchmark for the Monte Carlo price, and they give the delta
used to hedge. S can be a single number or a numpy array of prices.
"""
import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option="call"):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option="call"):
    d1, _ = d1_d2(S, K, T, r, sigma)
    if option == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def bs_vega(S, K, T, r, sigma):
    """Change in price for +1.00 of vol (divide by 100 for +1 vol point)."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)
