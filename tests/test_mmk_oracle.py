"""mmk1k_metrics 对照 Decimal(prec=60) 直接式 oracle 的数值正确性测试（规格 J1+N5）。"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from typing import cast

import pytest

from trl_sb3.env.mmk import MMKMetrics, mmk1k_metrics

CAPACITY = 10000
MUS = (1000, 2000, 3000)
# 三分支（rho<1 / ==1 / >1）+ 近 1 双侧；按 N§5：最靠近 1 的采样点用 1±1e-3 / 1±1e-5
# （|rho-1|<=1e-6 时浮点构造误差会放大相对误差），1.0 用 lam=mu 精确整除。
RHOS = (
    0.0, 0.001, 0.5, 0.9, 0.99, 0.999,
    1.0 - 1e-5, 1.0 - 1e-3, 1.0, 1.0 + 1e-5, 1.0 + 1e-3,
    1.001, 1.01, 2.0, 10.0, 1e3, 1e6,
)


def _oracle(lam: float, mu: int, k: int) -> tuple[float, float, float]:
    """Decimal prec=60 直接式 oracle（与 legacy routingEnv.py:425-450 同构公式）。"""
    if lam == 0.0:
        return 0.0, 0.0, 0.0
    with localcontext() as ctx:
        ctx.prec = 60
        rho = Decimal(lam) / Decimal(mu)
        if rho == 1:
            loss = Decimal(1) / (k + 1)
            length = Decimal(k) / 2
        else:
            d = 1 - rho ** (k + 1)
            loss = (1 - rho) * rho**k / d
            length = rho / (1 - rho) - (k + 1) * rho ** (k + 1) / d
        delay = length / (Decimal(lam) * (1 - loss))
        return float(loss), float(length), float(delay)


@pytest.mark.parametrize("mu", MUS)
@pytest.mark.parametrize("rho_target", RHOS)
def test_mmk_matches_decimal_oracle(rho_target: float, mu: int) -> None:
    """Given lam=rho_target*mu；When mmk1k_metrics；Then 与 oracle 一致（rel 1e-9）且全有限。"""
    lam = rho_target * mu  # rho_target=1.0 时 lam=mu 精确
    result = mmk1k_metrics(lam, mu, CAPACITY)
    assert math.isfinite(result.loss)
    assert math.isfinite(result.L)
    assert math.isfinite(result.delay)
    expected = _oracle(lam, mu, CAPACITY)
    for got, want in zip(cast(MMKMetrics, result), expected):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12), (
            f"rho={rho_target} mu={mu}: got={got} want={want}"
        )
