"""M/M/1/K 排队论纯函数（K=capacity=10000）——float64 数值稳定式。

legacy 出处：TRL_Routing/code/routingEnv.py:425-450（MMK 时延 / loss 丢包，Decimal 直接式）。
本实现按移植规格 I 节改写：rho>1 分支以 r=rho**-(K+1) 换元，防 rho**(K+1) 上溢
与 inf/inf=NaN；rho==1 取极限 loss=1/(K+1)、L=K/2；lam==0 短路 (0,0,0)。
大 rho 极限：delay→K/mu、loss→1-1/rho。
"""

from __future__ import annotations

from typing import NamedTuple


class MMKMetrics(NamedTuple):
    """M/M/1/K 单节点指标：丢包率 loss、平均队列长 L、平均时延 delay。"""

    loss: float
    L: float
    delay: float


def mmk1k_metrics(lam: float, mu: float, k: int = 10000) -> MMKMetrics:
    """M/M/1/K 指标（legacy routingEnv.py:425-450 的 float64 稳定式）。

    lam 到达率、mu 服务率、k 缓冲容量（论文 K=capacity=10000）。
    """
    if lam == 0:
        return MMKMetrics(0.0, 0.0, 0.0)
    rho = lam / mu
    if rho == 1.0:
        loss = 1.0 / (k + 1)
        queue_length = k / 2.0
    elif rho < 1.0:
        d = 1.0 - rho ** (k + 1)
        loss = (1.0 - rho) * rho**k / d
        queue_length = rho / (1.0 - rho) - (k + 1) * rho ** (k + 1) / d
    else:
        r = rho ** (-(k + 1))
        loss = (1.0 / rho - 1.0) / (r - 1.0)
        queue_length = rho / (1.0 - rho) - (k + 1) / (r - 1.0)
    delay = queue_length / (lam * (1.0 - loss))
    return MMKMetrics(loss, queue_length, delay)


def mmk_loss(lam: float, mu: float, k: int = 10000) -> float:
    """传播环用的节点丢包率（legacy routingEnv.py:441-450 loss）。"""
    return mmk1k_metrics(lam, mu, k).loss


def mmk_delay(lam: float, mu: float, k: int = 10000) -> float:
    """奖励用的节点平均时延（legacy routingEnv.py:425-439 MMK）。"""
    return mmk1k_metrics(lam, mu, k).delay
