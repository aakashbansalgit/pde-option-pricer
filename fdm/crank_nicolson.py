"""
Crank-Nicolson finite differences for the Black-Scholes PDE.

    dV/dt + 1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V = 0

solved backwards from the payoff on a uniform grid in S. One function prices
European and American calls and puts, and up or down knock-out barriers
monitored either continuously or on a set of dates.

Three choices matter more than they look:

- Rannacher start. Crank-Nicolson is second order but only neutrally stable
  for the highest-frequency error, so the kink in the payoff at the strike
  rings through the solution and spoils the Greeks. Two fully implicit
  steps, each taken as two half-steps, damp it. The same restart is applied
  after every discrete barrier date, since each knock-out is a fresh jump.

- Grid alignment. A continuous barrier has to sit exactly on a grid line, so
  the grid ends at the barrier; a barrier between nodes makes the error jump
  around as the grid is refined instead of falling. A barrier checked on
  discrete dates is the opposite case and belongs halfway between two nodes.

- American exercise. Taking the maximum of the continuation value and the
  payoff after each step (projection) is simple and only first-order
  accurate. For a put the early-exercise problem can instead be solved
  exactly at every step with the Brennan-Schwartz algorithm, a Thomas solve
  that applies the exercise constraint during the back substitution.
"""
from math import exp

import numpy as np


def _tridiag_solve(lower, diag, upper, rhs):
    """Thomas algorithm. lower[0] and upper[-1] are unused."""
    n = len(diag)
    c = np.empty(n)
    d = np.empty(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _brennan_schwartz_put(lower, diag, upper, rhs, payoff):
    """Exact solution of the tridiagonal linear complementarity problem for a
    put, whose exercise region is the low end of the grid. Eliminate from the
    top down, then substitute from the bottom up, taking the maximum with the
    payoff at each node as it is computed."""
    n = len(diag)
    c = np.empty(n)
    d = np.empty(n)
    c[-1] = lower[-1] / diag[-1]
    d[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        m = diag[i] - upper[i] * c[i + 1]
        c[i] = lower[i] / m
        d[i] = (rhs[i] - upper[i] * d[i + 1]) / m
    x = np.empty(n)
    x[0] = max(d[0], payoff[0])
    for i in range(1, n):
        x[i] = max(d[i] - c[i] * x[i - 1], payoff[i])
    return x


def price(S0, K, T, r, sigma, q=0.0, kind="call", american=False,
          exercise="brennan-schwartz", barrier=None, style="up-out",
          monitoring=None, M=400, N=400, rannacher=True, s_max=4.0,
          align_barrier=True, return_grid=False):
    """Price one option.

    barrier / style  knock-out level and direction ("up-out" or "down-out").
    monitoring       None for a continuous barrier, or a list of dates, in
                     years from today, on which the barrier is checked.
    exercise         "brennan-schwartz" (puts only) or "projection".
    align_barrier    put a continuous barrier on the edge of the grid. False
                     is only there to show what goes wrong without it.
    M, N             price and time steps.
    """
    continuous = barrier is not None and monitoring is None
    lo, hi = 0.0, s_max * max(S0, K)
    if continuous and align_barrier and style == "up-out":
        hi = barrier          # the barrier is the top grid line
    if continuous and align_barrier and style == "down-out":
        lo = barrier          # the barrier is the bottom grid line
    if barrier is not None and not continuous:
        # A barrier checked on dates should fall halfway between two nodes.
        # Knocking out at the node on the barrier itself biases the price
        # down, and the bias only shrinks at first order as the grid refines.
        k = max(1, round(barrier / hi * M - 0.5))
        hi = M * barrier / (k + 0.5)
    S = np.linspace(lo, hi, M + 1)
    dS = S[1] - S[0]

    payoff = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    V = payoff.copy()
    # knock out at expiry if the barrier is continuous or expiry is a date
    if continuous or (barrier is not None and any(abs(t - T) < 1e-12 for t in monitoring)):
        V[_knocked(S, barrier, style)] = 0.0

    j = S[1:-1] / dS
    alpha = 0.5 * sigma ** 2 * j ** 2
    beta = 0.5 * (r - q) * j
    lower_op = alpha - beta          # coefficient on V[i-1]
    diag_op = -2 * alpha - r         # on V[i]
    upper_op = alpha + beta          # on V[i+1]

    if american and kind == "call" and exercise == "brennan-schwartz":
        exercise = "projection"      # Brennan-Schwartz as written is for puts
    # Time grid in time-to-expiry. Every monitoring date has to be a grid
    # point: knocking out even one step early acts like an extra monitoring
    # date and adds an error that only shrinks with the square root of dt.
    taus = _time_grid(T, N, monitoring)
    knock_at = {round(T - t, 12) for t in (monitoring or []) if 0 <= T - t < T + 1e-12}

    implicit_left = 2 if rannacher else 0
    lcp = american and exercise == "brennan-schwartz"
    for tau_old, tau_new in zip(taus[:-1], taus[1:]):
        dt = tau_new - tau_old
        if implicit_left > 0:
            # an implicit step taken as two half-steps
            mid = tau_old + dt / 2
            V = _step(V, S, K, r, q, dt / 2, tau_old, mid, 1.0,
                      lower_op, diag_op, upper_op, kind, barrier, continuous, style)
            V = _step(V, S, K, r, q, dt / 2, mid, tau_new, 1.0,
                      lower_op, diag_op, upper_op, kind, barrier, continuous, style,
                      lcp, payoff)
            implicit_left -= 1
        else:
            V = _step(V, S, K, r, q, dt, tau_old, tau_new, 0.5,
                      lower_op, diag_op, upper_op, kind, barrier, continuous, style,
                      lcp, payoff)
        if american and exercise == "projection":
            V = np.maximum(V, payoff)
        if continuous:
            V[_knocked(S, barrier, style)] = 0.0
        if round(tau_new, 12) in knock_at:
            V[_knocked(S, barrier, style)] = 0.0
            # the knock-out puts a fresh jump in the solution, so smooth it
            # the same way as the payoff kink at expiry
            implicit_left = 2 if rannacher else 0

    value = float(np.interp(S0, S, V))
    return (value, S, V) if return_grid else value


def _time_grid(T, N, monitoring):
    """About N steps in time to expiry from 0 to T, with a node on every
    monitoring date and steps spread evenly within each interval."""
    cuts = sorted({0.0, T, *(round(T - t, 12) for t in (monitoring or []) if 0 < T - t < T)})
    nodes = [0.0]
    for a, b in zip(cuts[:-1], cuts[1:]):
        k = max(1, int(round(N * (b - a) / T)))
        nodes.extend(np.linspace(a, b, k + 1)[1:])
    return np.array(nodes)


def _knocked(S, H, style):
    return S >= H if style == "up-out" else S <= H


def _boundaries(S, K, r, q, tau, kind, barrier, continuous, style):
    """Values on the bottom and top grid lines at time to expiry tau."""
    lo, hi = S[0], S[-1]
    if kind == "call":
        v_lo = 0.0
        v_hi = hi * exp(-q * tau) - K * exp(-r * tau)
    else:
        v_lo = K * exp(-r * tau) - lo * exp(-q * tau)
        v_hi = 0.0
    if continuous and style == "up-out":
        v_hi = 0.0
    if continuous and style == "down-out":
        v_lo = 0.0
    return v_lo, v_hi


def _step(V, S, K, r, q, dt, tau_old, tau_new, theta, lo_op, di_op, up_op,
          kind, barrier, continuous, style, lcp=False, payoff=None):
    """One theta-scheme step from tau_old to tau_new (time to expiry)."""
    lo_old, hi_old = _boundaries(S, K, r, q, tau_old, kind, barrier, continuous, style)
    lo_new, hi_new = _boundaries(S, K, r, q, tau_new, kind, barrier, continuous, style)

    # (I - theta dt L) V_new = (I + (1 - theta) dt L) V_old
    e = (1 - theta) * dt
    rhs = V[1:-1] + e * (lo_op * V[:-2] + di_op * V[1:-1] + up_op * V[2:])
    i = theta * dt
    lower, diag, upper = -i * lo_op, 1 - i * di_op, -i * up_op
    # known boundary values on the new time level move to the right-hand side
    rhs[0] -= lower[0] * lo_new
    rhs[-1] -= upper[-1] * hi_new

    if lcp:
        inner = _brennan_schwartz_put(lower, diag, upper, rhs, payoff[1:-1])
        lo_new = max(lo_new, payoff[0])
    else:
        inner = _tridiag_solve(lower, diag, upper, rhs)
    return np.concatenate([[lo_new], inner, [hi_new]])
