"""A few sanity checks. Run with: python -m pytest -q"""
import numpy as np

from mchedge import bs_delta, bs_price, delta_hedge, derman_kamal_std, mc_price


def test_put_call_parity():
    call = bs_price(100, 105, 1, 0.03, 0.25, "call")
    put = bs_price(100, 105, 1, 0.03, 0.25, "put")
    assert np.isclose(call - put, 100 - 105 * np.exp(-0.03))


def test_delta_is_slope_of_price():
    h = 1e-4
    slope = (bs_price(100 + h, 105, 1, 0.03, 0.25) - bs_price(100 - h, 105, 1, 0.03, 0.25)) / (2 * h)
    assert np.isclose(bs_delta(100, 105, 1, 0.03, 0.25), slope, atol=1e-6)


def test_mc_close_to_black_scholes():
    for option in ["call", "put"]:
        price, se = mc_price(100, 105, 1, 0.03, 0.25, option, seed=1)
        assert abs(price - bs_price(100, 105, 1, 0.03, 0.25, option)) < 4 * se


def test_variance_reduction_helps():
    _, se_plain = mc_price(100, 105, 1, 0.03, 0.25, antithetic=False, control_variate=False, seed=2)
    _, se_best = mc_price(100, 105, 1, 0.03, 0.25, seed=2)
    assert se_best < 0.5 * se_plain


def test_hedge_error_matches_derman_kamal():
    pnl, _, _ = delta_hedge(n_rebal=252, seed=4)
    assert abs(pnl.mean()) < 0.05
    assert 0.85 < pnl.std() / derman_kamal_std(100, 100, 1, 0.02, 0.2, 252) < 1.15


def test_hedge_error_falls_as_one_over_sqrt_n():
    pnl_50, _, _ = delta_hedge(n_rebal=50, seed=5)
    pnl_800, _, _ = delta_hedge(n_rebal=800, seed=6)
    assert 3.4 < pnl_50.std() / pnl_800.std() < 4.6  # theory: sqrt(800 / 50) = 4


def test_selling_vol_too_cheap_loses_money():
    pnl, _, _ = delta_hedge(sigma_real=0.30, sigma_implied=0.20, n_rebal=252, n_paths=10_000, seed=7)
    expected = bs_price(100, 100, 1, 0.02, 0.2) - bs_price(100, 100, 1, 0.02, 0.3)
    assert abs(pnl.mean() - expected) < 0.15 * abs(expected)


def test_costs_reduce_pnl_by_their_amount():
    pnl_free, _, _ = delta_hedge(n_rebal=52, n_paths=5_000, seed=8)
    pnl_tc, costs, _ = delta_hedge(n_rebal=52, n_paths=5_000, seed=8, tc=0.002)
    assert np.isclose(pnl_free.mean() - pnl_tc.mean(), costs.mean())


def test_put_hedge_works():
    pnl, _, premium = delta_hedge(option="put", n_rebal=252, n_paths=10_000, seed=9)
    assert pnl.std() < 0.2 * premium
