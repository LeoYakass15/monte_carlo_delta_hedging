"""Discrete delta hedging of a short option.

The trader sells the option at the Black-Scholes price (implied vol), buys delta
shares and puts the rest of the cash in the bank at rate r. At N equally spaced
dates the hedge is rebalanced to the new Black-Scholes delta. The stock moves
with the realised vol, which can differ from the implied vol.

At maturity: P&L = cash + shares held * S_T - option payoff (discounted to today).
With continuous hedging and the right vol this P&L would be exactly 0.
"""
import numpy as np

from .black_scholes import bs_delta, bs_price, bs_vega


def delta_hedge(S0=100.0, K=100.0, T=1.0, r=0.02, sigma_real=0.2, sigma_implied=None,
                n_rebal=52, n_paths=20_000, option="call", tc=0.0, seed=None):
    """Simulate the hedge on n_paths paths.

    tc is the proportional transaction cost (0.001 = 10 bp of the traded amount).
    Return (pnl per path, transaction costs per path, premium received).
    """
    if sigma_implied is None:
        sigma_implied = sigma_real
    rng = np.random.default_rng(seed)
    dt = T / n_rebal

    # t = 0: sell the option and buy the first hedge
    S = np.full(n_paths, S0)
    premium = bs_price(S0, K, T, r, sigma_implied, option)
    delta = bs_delta(S, K, T, r, sigma_implied, option)
    costs = tc * np.abs(delta) * S
    cash = premium - delta * S - costs

    for i in range(1, n_rebal + 1):
        # the stock moves, the cash earns interest
        Z = rng.standard_normal(n_paths)
        S = S * np.exp((r - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * Z)
        cash = cash * np.exp(r * dt)
        costs = costs * np.exp(r * dt)

        # rebalance (no trade on the last date: the option expires)
        if i < n_rebal:
            new_delta = bs_delta(S, K, T - i * dt, r, sigma_implied, option)
            trade = new_delta - delta
            cost = tc * np.abs(trade) * S
            cash = cash - trade * S - cost
            costs = costs + cost
            delta = new_delta

    if option == "call":
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)
    pnl = np.exp(-r * T) * (cash + delta * S - payoff)
    return pnl, np.exp(-r * T) * costs, premium


def derman_kamal_std(S0, K, T, r, sigma, n_rebal):
    """Approximate std dev of the hedging error (Derman & Kamal, 1999):
    sqrt(pi / 4) * vega * sigma / sqrt(N)."""
    return np.sqrt(np.pi / 4) * bs_vega(S0, K, T, r, sigma) * sigma / np.sqrt(n_rebal)
