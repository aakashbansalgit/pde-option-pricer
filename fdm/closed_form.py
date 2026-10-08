"""
Closed-form prices used to check the finite difference solver.

Black-Scholes with a continuous dividend yield, the standard single-barrier
formulas (Reiner and Rubinstein, 1991, as set out in Hull), and the
Broadie-Glasserman-Kou shift that turns a continuous-barrier price into a
good approximation for a barrier checked only on discrete dates.
"""
from math import exp, log, sqrt

from scipy.stats import norm

N = norm.cdf


def bs(S, K, T, r, sigma, q=0.0, kind="call"):
    if T <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    s = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / s
    d2 = d1 - s
    if kind == "call":
        return S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)
    return K * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)


def barrier(S, K, H, T, r, sigma, q=0.0, kind="call", style="up-out"):
    """Continuously monitored single barrier, no rebate.

    style is one of up-out, up-in, down-out, down-in. Knock-in prices come from
    in-out parity, which holds exactly: knock-in plus knock-out is the vanilla."""
    vanilla = bs(S, K, T, r, sigma, q, kind)
    direction, knock = style.split("-")
    if (direction == "up" and S >= H) or (direction == "down" and S <= H):
        out = 0.0
    else:
        out = _knock_out(S, K, H, T, r, sigma, q, kind, direction)
    return out if knock == "out" else vanilla - out


def _knock_out(S, K, H, T, r, sigma, q, kind, direction):
    st = sigma * sqrt(T)
    lam = (r - q + 0.5 * sigma ** 2) / sigma ** 2
    x1 = log(S / H) / st + lam * st
    y = log(H * H / (S * K)) / st + lam * st
    y1 = log(H / S) / st + lam * st
    a = S * exp(-q * T)
    b = K * exp(-r * T)
    p2l = (H / S) ** (2 * lam)
    p2l2 = (H / S) ** (2 * lam - 2)
    vanilla = bs(S, K, T, r, sigma, q, kind)

    if kind == "call" and direction == "up":
        if H <= K:
            return 0.0
        cui = (a * N(x1) - b * N(x1 - st) - a * p2l * (N(-y) - N(-y1))
               + b * p2l2 * (N(-y + st) - N(-y1 + st)))
        return vanilla - cui
    if kind == "call" and direction == "down":
        if H <= K:
            cdi = a * p2l * N(y) - b * p2l2 * N(y - st)
            return vanilla - cdi
        return (a * N(x1) - b * N(x1 - st) - a * p2l * N(y1)
                + b * p2l2 * N(y1 - st))
    if kind == "put" and direction == "down":
        if H >= K:
            return 0.0
        pdi = (-a * N(-x1) + b * N(-x1 + st) + a * p2l * (N(y) - N(y1))
               - b * p2l2 * (N(y - st) - N(y1 - st)))
        return vanilla - pdi
    if kind == "put" and direction == "up":
        if H >= K:
            pui = -a * p2l * N(-y) + b * p2l2 * N(-y + st)
            return vanilla - pui
        return (-a * N(-x1) + b * N(-x1 + st) + a * p2l * N(-y1)
                - b * p2l2 * N(-y1 + st))
    raise ValueError(kind, direction)


BGK = 0.5826   # -zeta(1/2) / sqrt(2 pi)


def barrier_discrete(S, K, H, T, r, sigma, n_dates, q=0.0, kind="call", style="up-out"):
    """Broadie, Glasserman and Kou (1997): a barrier checked on n equally
    spaced dates behaves like a continuous barrier moved away from the spot
    by a factor exp(0.5826 sigma sqrt(T / n))."""
    shift = exp(BGK * sigma * sqrt(T / n_dates))
    H_eff = H * shift if style.startswith("up") else H / shift
    return barrier(S, K, H_eff, T, r, sigma, q, kind, style)
