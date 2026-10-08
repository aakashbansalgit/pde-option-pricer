"""Checks the solver against closed forms, a binomial tree and quadrature.

Run from the repo root: python -m pytest tests  (or python tests/test_fdm.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdm import closed_form as cf
from fdm import crank_nicolson as cn
import study


def test_european_matches_black_scholes():
    for kind in ("call", "put"):
        for S in (80.0, 100.0, 125.0):
            pde = cn.price(S, 100.0, 1.0, 0.05, 0.2, q=0.02, kind=kind, M=800, N=800)
            assert abs(pde - cf.bs(S, 100.0, 1.0, 0.05, 0.2, 0.02, kind)) < 2e-3, (kind, S)


def test_second_order_convergence():
    exact = cf.bs(100.0, 100.0, 1.0, 0.05, 0.2)
    e1 = abs(cn.price(100.0, 100.0, 1.0, 0.05, 0.2, M=200, N=200) - exact)
    e2 = abs(cn.price(100.0, 100.0, 1.0, 0.05, 0.2, M=400, N=400) - exact)
    assert 3.5 < e1 / e2 < 4.5


def test_continuous_barriers_match_closed_form():
    cases = [("call", 120.0, "up-out", 0.16), ("put", 85.0, "down-out", 0.2)]
    for kind, H, style, s in cases:
        pde = cn.price(100.0, 100.0, 1.0, 0.05, s, kind=kind, barrier=H, style=style, M=800, N=800)
        exact = cf.barrier(100.0, 100.0, H, 1.0, 0.05, s, kind=kind, style=style)
        assert abs(pde - exact) < 2e-3, (kind, style, pde, exact)


def test_american_put_against_binomial_tree():
    ref = 0.5 * (study.crr_american_put(100.0, 100.0, 1.0, 0.05, 0.16, 4000)
                 + study.crr_american_put(100.0, 100.0, 1.0, 0.05, 0.16, 4001))
    am = cn.price(100.0, 100.0, 1.0, 0.05, 0.16, kind="put", american=True, M=800, N=800)
    eu = cf.bs(100.0, 100.0, 1.0, 0.05, 0.16, kind="put")
    assert abs(am - ref) < 3e-3 and am > eu + 0.4


def test_discrete_monitoring_against_quadrature():
    for n in (4, 12):
        dates = list(np.arange(1, n + 1) / n)
        pde = cn.price(100.0, 100.0, 1.0, 0.05, 0.16, barrier=120.0, style="up-out",
                       monitoring=dates, M=1600, N=1600)
        quad = study.quadrature_up_out(100.0, 100.0, 120.0, 1.0, 0.05, 0.16, n)
        assert abs(pde - quad) < 3e-3, (n, pde, quad)


def test_knock_in_by_in_out_parity():
    S, K, H, T, r, s = 5650.38, 5600.0, 5500.0, 0.21, 0.0433, 0.17
    vanilla = cn.price(S, K, T, r, s, kind="put", M=1600, N=800)
    out = cn.price(S, K, T, r, s, kind="put", barrier=H, style="down-out", M=1600, N=800)
    assert abs((vanilla - out) - cf.barrier(S, K, H, T, r, s, kind="put", style="down-in")) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
