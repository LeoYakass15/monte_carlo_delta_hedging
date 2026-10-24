"""Monte Carlo price of a European option.

Under the risk-neutral measure the price at maturity is
    S_T = S0 * exp((r - sigma^2 / 2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)
so we draw Z, compute the discounted payoff on each path and take the average.

Two classic tricks reduce the noise:
- antithetic variates: every draw Z is paired with -Z;
- control variate: the discounted stock price e^(-rT) * S_T has a known mean (S0),
  so we use its sampling error to correct the payoff average.
"""
import numpy as np


def mc_price(S0, K, T, r, sigma, option="call", n_paths=200_000,
             antithetic=True, control_variate=True, seed=None):
    """Return (price, standard error)."""
    rng = np.random.default_rng(seed)

    if antithetic:
        Z = rng.standard_normal(n_paths // 2)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(n_paths)

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option == "call":
        payoff = np.exp(-r * T) * np.maximum(ST - K, 0)
    else:
        payoff = np.exp(-r * T) * np.maximum(K - ST, 0)
    stock = np.exp(-r * T) * ST  # control variate, its true mean is S0

    if antithetic:
        # Z and -Z are not independent, so the independent samples are the pair averages
        half = len(Z) // 2
        payoff = (payoff[:half] + payoff[half:]) / 2
        stock = (stock[:half] + stock[half:]) / 2

    if control_variate:
        beta = np.cov(payoff, stock)[0, 1] / np.var(stock, ddof=1)
        payoff = payoff - beta * (stock - S0)

    price = payoff.mean()
    std_error = payoff.std(ddof=1) / np.sqrt(len(payoff))
    return price, std_error
