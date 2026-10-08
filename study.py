"""
Convergence study for the Crank-Nicolson solver.

    python study.py

Each experiment checks the solver against an independent answer, a closed
form, a 20,000-step binomial tree or Monte Carlo, and writes a table to
results/ and a chart to figures/.
"""
from math import exp, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fdm import closed_form as cf
from fdm import crank_nicolson as cn

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"
GRIDS = [50, 100, 200, 400, 800, 1600]


def order(errors):
    """Observed order of convergence from successive halvings of the step."""
    e = np.abs(np.asarray(errors, float))
    return [np.nan] + list(np.log2(e[:-1] / e[1:]))


def european():
    S, K, T, r, s = 100.0, 100.0, 1.0, 0.05, 0.2
    exact = cf.bs(S, K, T, r, s, kind="call")
    rows = []
    for M in GRIDS:
        rows.append({"M=N": M,
                     "error": cn.price(S, K, T, r, s, M=M, N=M) - exact,
                     "error_no_rannacher": cn.price(S, K, T, r, s, M=M, N=M, rannacher=False) - exact})
    df = pd.DataFrame(rows)
    df["order"] = order(df["error"])
    df.to_csv(RES / "european_convergence.csv", index=False, float_format="%.3e")

    # Gamma near the strike a week before expiry, on a fine price grid with
    # large time steps. Plain Crank-Nicolson rings around the kink in the
    # payoff; the Rannacher start removes it.
    fig, ax = plt.subplots(figsize=(8, 4))
    for ran, label in ((False, "Crank-Nicolson"), (True, "with Rannacher start")):
        _, grid, V = cn.price(S, K, 7 / 365, r, s, M=1600, N=4, rannacher=ran, return_grid=True)
        gamma = np.gradient(np.gradient(V, grid), grid)
        keep = (grid > 85) & (grid < 115)
        ax.plot(grid[keep], gamma[keep], label=label)
    ax.set_xlabel("spot")
    ax.set_ylabel("gamma")
    ax.set_title("Call gamma a week before expiry, 1,600 price steps, 4 time steps")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "gamma_rannacher.png", dpi=120)
    return df


def barrier_alignment():
    S, K, H, T, r, s = 100.0, 100.0, 120.0, 1.0, 0.05, 0.16
    exact = cf.barrier(S, K, H, T, r, s, kind="call", style="up-out")
    rows = []
    for M in GRIDS:
        on = cn.price(S, K, T, r, s, barrier=H, style="up-out", M=M, N=M)
        off = cn.price(S, K, T, r, s, barrier=H, style="up-out", M=M, N=M, align_barrier=False)
        rows.append({"M=N": M, "error_on_grid": on - exact, "error_off_grid": off - exact})
    df = pd.DataFrame(rows)
    df["order_on_grid"] = order(df["error_on_grid"])
    df.to_csv(RES / "barrier_alignment.csv", index=False, float_format="%.3e")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(df["M=N"], df["error_on_grid"].abs(), "o-", label="barrier on the grid edge")
    ax.loglog(df["M=N"], df["error_off_grid"].abs(), "s-", label="barrier between nodes")
    ax.set_xlabel("grid points in price and time")
    ax.set_ylabel("absolute error")
    ax.set_title("Up-and-out call, continuous barrier at 120")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "barrier_alignment.png", dpi=120)
    return df, exact


def crr_american_put(S, K, T, r, s, n):
    dt = T / n
    u = exp(s * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    disc = exp(-r * dt)
    j = np.arange(n + 1)
    V = np.maximum(K - S * u ** (n - j) * d ** j, 0.0)
    for i in range(n - 1, -1, -1):
        j = np.arange(i + 1)
        V = np.maximum(disc * (p * V[:-1] + (1 - p) * V[1:]), K - S * u ** (i - j) * d ** j)
    return V[0]


def american():
    S, K, T, r, s = 100.0, 100.0, 1.0, 0.05, 0.16
    # averaging two adjacent step counts removes most of the tree's odd-even wobble
    ref = 0.5 * (crr_american_put(S, K, T, r, s, 20000) + crr_american_put(S, K, T, r, s, 20001))
    rows = []
    for M in GRIDS[:-1]:
        bs_ = cn.price(S, K, T, r, s, kind="put", american=True, M=M, N=M)
        pr = cn.price(S, K, T, r, s, kind="put", american=True, exercise="projection", M=M, N=M)
        rows.append({"M=N": M, "error_brennan_schwartz": bs_ - ref, "error_projection": pr - ref})
    df = pd.DataFrame(rows)
    df["order_brennan_schwartz"] = order(df["error_brennan_schwartz"])
    df["order_projection"] = order(df["error_projection"])
    df.to_csv(RES / "american_put.csv", index=False, float_format="%.3e")

    # early-exercise boundary: the highest spot at which exercising is optimal
    taus = np.linspace(0.02, 1.0, 50)
    boundary = []
    for tau in taus:
        _, grid, V = cn.price(S, K, tau, r, s, kind="put", american=True, M=800,
                              N=max(20, int(400 * tau)), return_grid=True)
        exercised = grid[(V <= np.maximum(K - grid, 0) + 1e-9) & (grid < K)]
        boundary.append(exercised.max() if len(exercised) else np.nan)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(taus, boundary)
    ax.set_xlabel("time to expiry, years")
    ax.set_ylabel("exercise below this spot")
    ax.set_title("American put early-exercise boundary, K = 100")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "exercise_boundary.png", dpi=120)
    return df, ref


def mc_up_out(S, K, H, T, r, s, n_dates, paths=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_dates
    total = total2 = 0.0
    for _ in range(paths // 100_000):
        z = rng.standard_normal((100_000, n_dates))
        lp = np.log(S) + np.cumsum((r - 0.5 * s * s) * dt + s * sqrt(dt) * z, axis=1)
        pay = exp(-r * T) * (lp.max(axis=1) < np.log(H)) * np.maximum(np.exp(lp[:, -1]) - K, 0)
        total += pay.sum()
        total2 += (pay ** 2).sum()
    m = total / paths
    return m, sqrt(total2 / paths - m * m) / sqrt(paths)


def quadrature_up_out(S, K, H, T, r, s, n_dates, points=6001):
    """Exact price for a barrier checked on n equally spaced dates, by
    integrating the lognormal transition density from one date to the next
    over the surviving region below the barrier."""
    from scipy.stats import norm
    dt = T / n_dates
    mu, sd = (r - 0.5 * s * s) * dt, s * sqrt(dt)
    x = np.linspace(np.log(S) - 8 * s * sqrt(T), np.log(H), points)
    dx = x[1] - x[0]
    V = np.maximum(np.exp(x) - K, 0.0)
    V[-1] *= 0.5
    kernel = norm.pdf(x[None, :] - x[:, None] - mu, scale=sd) * dx
    for _ in range(n_dates - 1):
        V = kernel @ V
    w = norm.pdf(x - np.log(S) - mu, scale=sd) * dx
    return exp(-r * T) * w @ V


def discrete_monitoring():
    S, K, H, T, r, s = 100.0, 100.0, 120.0, 1.0, 0.05, 0.16
    rows = []
    for n in (1, 4, 12, 52, 252):
        dates = list(np.arange(1, n + 1) / n)
        pde = cn.price(S, K, T, r, s, barrier=H, style="up-out", monitoring=dates,
                       M=1600, N=max(1600, 4 * n))
        quad = quadrature_up_out(S, K, H, T, r, s, n) if n <= 52 else np.nan
        bgk = cf.barrier_discrete(S, K, H, T, r, s, n, kind="call", style="up-out")
        mc, se = mc_up_out(S, K, H, T, r, s, n)
        rows.append({"dates": n, "pde": pde, "quadrature": quad, "monte_carlo": mc,
                     "mc_se": se, "bgk_approx": bgk})
    rows.append({"dates": "continuous",
                 "pde": cn.price(S, K, T, r, s, barrier=H, style="up-out", M=1600, N=1600),
                 "quadrature": np.nan, "monte_carlo": np.nan, "mc_se": np.nan,
                 "bgk_approx": cf.barrier(S, K, H, T, r, s, kind="call", style="up-out")})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "discrete_monitoring.csv", index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(7, 4))
    d = df[df["dates"] != "continuous"]
    n = d["dates"].astype(int)
    ax.semilogx(n, d["pde"], "o-", label="Crank-Nicolson")
    ax.errorbar(n, d["monte_carlo"], yerr=2 * d["mc_se"], fmt="none", ecolor="black", capsize=4,
                label="Monte Carlo, 2 s.e.")
    ax.semilogx(n, d["bgk_approx"], "s--", label="continuity correction (BGK)")
    ax.axhline(df["pde"].iloc[-1], color="grey", ls=":", label="continuous barrier")
    ax.set_xlabel("monitoring dates per year")
    ax.set_ylabel("up-and-out call price")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "discrete_monitoring.png", dpi=120)
    return df


def spx_down_and_in_put():
    """A down-and-in put on the S&P 500 with the barrier just below spot.
    The PDE prices the knock-out and the vanilla; the knock-in is their
    difference by in-out parity."""
    S, K, H, T, r, s = 5650.38, 5600.0, 5500.0, 0.21, 0.0433, 0.17
    vanilla = cn.price(S, K, T, r, s, kind="put", M=1600, N=800)
    knock_out = cn.price(S, K, T, r, s, kind="put", barrier=H, style="down-out", M=1600, N=800)
    exact = cf.barrier(S, K, H, T, r, s, kind="put", style="down-in")

    rng = np.random.default_rng(1)
    n, paths = 2000, 200_000
    dt = T / n
    z = rng.standard_normal((paths, n))
    lp = np.log(S) + np.cumsum((r - 0.5 * s * s) * dt + s * sqrt(dt) * z, axis=1)
    pay = exp(-r * T) * (lp.min(axis=1) <= np.log(H)) * np.maximum(K - np.exp(lp[:, -1]), 0)
    df = pd.DataFrame([{
        "vanilla_put_pde": vanilla, "down_out_put_pde": knock_out,
        "down_in_put_pde": vanilla - knock_out, "down_in_put_closed_form": exact,
        "down_in_put_mc_2000_dates": pay.mean(), "mc_se": pay.std() / sqrt(paths),
    }])
    df.T.to_csv(RES / "spx_down_and_in_put.csv", header=["value"], float_format="%.4f")
    return df


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    print("European call\n", european(), "\n")
    df, exact = barrier_alignment()
    print(f"Up-and-out call, exact {exact:.6f}\n", df, "\n")
    df, ref = american()
    print(f"American put, binomial reference {ref:.6f}\n", df, "\n")
    print("Discrete monitoring\n", discrete_monitoring(), "\n")
    print("SPX down-and-in put\n", spx_down_and_in_put().T)


if __name__ == "__main__":
    main()
