"""Run all the studies and save the charts and tables in results/.

    python run_analysis.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mchedge import bs_price, delta_hedge, derman_kamal_std, mc_price

# Base case: 1-year at-the-money call
S0, K, T, r, SIGMA = 100.0, 100.0, 1.0, 0.02, 0.20
N_PATHS = 20_000   # paths per hedging simulation
TC = 0.001         # transaction cost: 10 bp
SEED = 42          # fixed seed, so the results are reproducible
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Monte Carlo pricer
# ---------------------------------------------------------------------------
print("== 1. Monte Carlo pricer ==")
bs = bs_price(S0, K, T, r, SIGMA)
price, se = mc_price(S0, K, T, r, SIGMA, n_paths=500_000, seed=SEED)
print(f"Black-Scholes: {bs:.4f}   Monte Carlo: {price:.4f} +/- {1.96 * se:.4f}")

# Convergence: standard error vs number of paths, for each method
methods = {
    "standard": dict(antithetic=False, control_variate=False),
    "antithetic": dict(antithetic=True, control_variate=False),
    "control variate": dict(antithetic=False, control_variate=True),
    "antithetic + control": dict(antithetic=True, control_variate=True),
}
rows = []
for n in [1_000, 4_000, 16_000, 64_000, 256_000]:
    for name, kw in methods.items():
        p, e = mc_price(S0, K, T, r, SIGMA, n_paths=n, seed=SEED, **kw)
        rows.append({"n_paths": n, "method": name, "price": p, "std_error": e, "bs_price": bs})
conv = pd.DataFrame(rows)
conv.to_csv(f"{OUT}/mc_convergence.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4.2))
for name, g in conv.groupby("method", sort=False):
    ax.loglog(g["n_paths"], g["std_error"], "o-", label=name)
ax.set(xlabel="Number of paths", ylabel="Standard error of the price",
       title="Monte Carlo: error vs number of paths")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
save(fig, "01_mc_convergence.png")

# Variance reduction = (std error of standard MC / std error of the method)^2
table = conv.pivot(index="n_paths", columns="method", values="std_error")
print("Variance reduction vs standard MC:")
for name in methods:
    gain = (table["standard"] ** 2 / table[name] ** 2).mean()
    print(f"  {name:<22} x{gain:.1f}")

# Check against Black-Scholes on 60 options
rows = []
i = 0
for option in ["call", "put"]:
    for mat in [0.25, 1.0]:
        for sig in [0.10, 0.20, 0.40]:
            for strike in [80, 90, 100, 110, 120]:
                p, e = mc_price(S0, strike, mat, r, sig, option, n_paths=200_000, seed=SEED + i)
                ref = bs_price(S0, strike, mat, r, sig, option)
                z = (p - ref) / e if e > 0 else 0.0  # gap measured in standard errors
                rows.append({"option": option, "K": strike, "T": mat, "sigma": sig, "bs_price": ref,
                             "mc_price": p, "std_error": e, "z": z, "bs_in_ci95": abs(z) < 1.96})
                i += 1
val = pd.DataFrame(rows)
val.to_csv(f"{OUT}/mc_vs_bs.csv", index=False)
n_ok = val["bs_in_ci95"].sum()
print(f"BS price inside the MC 95% interval for {n_ok} of {len(val)} options")
print(f"Mean z = {val['z'].mean():+.2f} (should be ~0), std of z = {val['z'].std():.2f} (should be ~1)")
print(f"Largest price gap: {(val['mc_price'] - val['bs_price']).abs().max():.4f}")

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.axhspan(-1.96, 1.96, alpha=0.15, label="95% interval")
for option, g in val.groupby("option"):
    ax.plot(g.index, g["z"], "o", label=f"{option}s")
ax.set(xlabel="Option number (60 options: 5 strikes x 3 vols x 2 maturities x call/put)",
       ylabel="(MC - BS) / std error",
       title=f"MC vs Black-Scholes: BS inside the 95% interval for {n_ok} of {len(val)} options")
ax.legend()
save(fig, "01b_mc_vs_bs.png")

# ---------------------------------------------------------------------------
# 2. Hedging error vs volatility and rebalancing frequency (implied = realised vol)
# ---------------------------------------------------------------------------
print("\n== 2. Hedging error vs vol and number of rebalancing dates N ==")
sigmas = [0.10, 0.20, 0.30, 0.40, 0.60]
freqs = [4, 12, 26, 52, 104, 252, 504, 1008]
rows = []
for i, sig in enumerate(sigmas):
    for j, n in enumerate(freqs):
        pnl, costs, premium = delta_hedge(S0, K, T, r, sig, n_rebal=n, n_paths=N_PATHS,
                                          seed=SEED + 1000 * i + j)
        rows.append({"sigma": sig, "n_rebal": n, "premium": premium,
                     "mean_pnl": pnl.mean(), "std_pnl": pnl.std(),
                     "std_pct_premium": 100 * pnl.std() / premium,
                     "derman_kamal_std": derman_kamal_std(S0, K, T, r, sig, n)})
grid = pd.DataFrame(rows)
grid.to_csv(f"{OUT}/hedging_grid.csv", index=False)

# Theory: std ~ 1/sqrt(N), so the slope of log(std) vs log(N) should be -0.5
for sig, g in grid.groupby("sigma"):
    slope = np.polyfit(np.log(g["n_rebal"]), np.log(g["std_pnl"]), 1)[0]
    print(f"  sigma = {sig:.0%}: slope {slope:.3f}")
ratio = grid["std_pnl"] / grid["derman_kamal_std"]
print(f"Simulated / Derman-Kamal: mean {ratio.mean():.2f}, min {ratio.min():.2f}, max {ratio.max():.2f}")

fig, ax = plt.subplots(figsize=(7, 4.6))
for sig, g in grid.groupby("sigma"):
    line, = ax.loglog(g["n_rebal"], g["std_pnl"], "o", label=f"σ = {sig:.0%}")
    ax.loglog(g["n_rebal"], g["derman_kamal_std"], "-", color=line.get_color())
ax.set(xlabel="Number of rebalancing dates N", ylabel="Std dev of the hedging P&L",
       title="Hedging error vs rebalancing frequency (dots: simulation, lines: theory)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
save(fig, "02_std_vs_rebal.png")

fig, ax = plt.subplots(figsize=(7, 4.2))
for n, g in grid.groupby("n_rebal"):
    ax.plot(g["sigma"] * 100, g["std_pnl"], "o-", label=f"N = {n}")
ax.set(xlabel="Volatility (%)", ylabel="Std dev of the hedging P&L",
       title="Hedging error vs volatility")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2)
save(fig, "03_std_vs_sigma.png")

heat = grid.pivot(index="sigma", columns="n_rebal", values="std_pct_premium")
fig, ax = plt.subplots(figsize=(7.5, 4.6))
im = ax.imshow(heat.values, cmap="Blues", aspect="auto", origin="lower")
ax.set_xticks(range(len(freqs)), freqs)
ax.set_yticks(range(len(sigmas)), [f"{s:.0%}" for s in sigmas])
for a in range(len(sigmas)):
    for b in range(len(freqs)):
        v = heat.values[a, b]
        ax.text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.55 * heat.values.max() else "black")
ax.set(xlabel="Number of rebalancing dates N", ylabel="Volatility",
       title="Hedging error as % of the option premium")
fig.colorbar(im, ax=ax, label="% of premium")
save(fig, "04_heatmap_rel_std.png")

# P&L distribution for monthly, weekly and daily hedging
fig, ax = plt.subplots(figsize=(7, 4.2))
bins = np.linspace(-6, 6, 90)
for n in [12, 52, 252]:
    pnl, _, _ = delta_hedge(S0, K, T, r, SIGMA, n_rebal=n, n_paths=N_PATHS, seed=SEED)
    ax.hist(pnl, bins=bins, density=True, histtype="step", lw=2, label=f"N = {n} (std = {pnl.std():.2f})")
ax.axvline(0, color="grey", ls=":")
ax.set(xlabel="Hedging P&L", ylabel="Density", title="Distribution of the hedging P&L")
ax.legend()
save(fig, "05_pnl_distributions.png")

# ---------------------------------------------------------------------------
# 3. Realised vol different from implied vol
# ---------------------------------------------------------------------------
print("\n== 3. Realised vol != implied vol (sold at 20%) ==")
rows = []
for i, sig_real in enumerate(np.linspace(0.10, 0.30, 9)):
    pnl, _, _ = delta_hedge(S0, K, T, r, sig_real, sigma_implied=SIGMA, n_rebal=252,
                            n_paths=N_PATHS, seed=SEED + i)
    theory = bs_price(S0, K, T, r, SIGMA) - bs_price(S0, K, T, r, sig_real)
    rows.append({"sigma_real": sig_real, "mean_pnl": pnl.mean(), "theory_mean": theory,
                 "std_pnl": pnl.std()})
mism = pd.DataFrame(rows)
mism.to_csv(f"{OUT}/vol_mismatch.csv", index=False)
print(mism.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.2))
x = mism["sigma_real"] * 100
ax.fill_between(x, mism["mean_pnl"] - mism["std_pnl"], mism["mean_pnl"] + mism["std_pnl"],
                alpha=0.15, label="± 1 std dev")
ax.plot(x, mism["mean_pnl"], "o", label="Simulated mean P&L")
ax.plot(x, mism["theory_mean"], "-", label="BS(20%) − BS(realised vol)")
ax.axhline(0, color="grey", ls=":")
ax.axvline(SIGMA * 100, color="grey", ls="--")
ax.set(xlabel="Realised volatility (%)", ylabel="P&L of the option seller",
       title="Option sold at 20% vol: P&L vs realised vol")
ax.legend()
save(fig, "06_vol_mismatch.png")

# ---------------------------------------------------------------------------
# 4. Transaction costs: hedging more often reduces risk but costs more
# ---------------------------------------------------------------------------
print(f"\n== 4. Transaction costs ({TC:.2%}) ==")
rows = []
for j, n in enumerate(freqs):
    pnl, costs, _ = delta_hedge(S0, K, T, r, SIGMA, n_rebal=n, n_paths=N_PATHS, tc=TC, seed=SEED + j)
    rows.append({"n_rebal": n, "std_pnl": pnl.std(), "mean_costs": costs.mean(),
                 "costs_plus_risk": costs.mean() + pnl.std()})
tc = pd.DataFrame(rows)
tc.to_csv(f"{OUT}/tc_tradeoff.csv", index=False)
print(tc.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogx(tc["n_rebal"], tc["std_pnl"], "o-", label="Risk (P&L std dev)")
ax.semilogx(tc["n_rebal"], tc["mean_costs"], "o-", label="Average costs")
ax.semilogx(tc["n_rebal"], tc["costs_plus_risk"], "o-", label="Costs + risk")
best = tc.loc[tc["costs_plus_risk"].idxmin()]
ax.annotate(f"best ≈ N = {int(best['n_rebal'])}", (best["n_rebal"], best["costs_plus_risk"]),
            xytext=(0, 12), textcoords="offset points", ha="center")
ax.set(xlabel="Number of rebalancing dates N", ylabel="Amount (same unit as the premium)",
       title="Risk vs transaction costs (10 bp)")
ax.grid(True, alpha=0.3)
ax.legend()
save(fig, "07_tc_tradeoff.png")

print(f"\nCharts and tables saved in {OUT}/")
