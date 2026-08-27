"""预注册指标（M3，计划 §5 指标三元组）：纯函数、无 IO、无 pandas。

三元组：① τ = 到达阈值 θ 的回合数（贪心评估曲线线性插值过阈，窗口末删失）；
② 固定窗口 W 归一化 AUC（梯形积分 ÷ W）；③ 渐近性能 = 末 k 个评估点均值
±95%CI（t 分布）。统计：配对 t / Wilcoxon（scipy.stats）+ 自实现 Holm 校正
——scipy.stats 的 false_discovery_control 是 BH/BY 族而非 Holm，故按定义手写。
θ/W/k 的取值钉在 config/metrics_prereg.yaml（预注册，防 p-hacking），本模块
不读配置、不含论文超参。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from math import copysign, inf, sqrt
from statistics import mean, stdev
from typing import Protocol, runtime_checkable

from scipy import stats

_CI_LEVEL = 0.95


@runtime_checkable
class _HasPValue(Protocol):
    """scipy wilcoxon 经 _axis_nan_policy_factory 装饰后静态返回类型被擦除（类型检查器见 `_`）；
    以结构协议锚定结果对象的 .pvalue 契约（WilcoxonResult namedtuple 的既有字段），
    isinstance（runtime_checkable → hasattr 检查）在静态与运行时两侧都成立。"""

    pvalue: float


def time_to_threshold(
    episodes: Sequence[int], values: Sequence[float], theta: float, window: int
) -> tuple[float, bool]:
    """贪心评估曲线上首个过阈 θ 的回合号（计划 §5-1）。

    插值语义：相邻曲线点 (e_i, v_i), (e_{i+1}, v_{i+1}) 跨越 θ
    （v_i < θ ≤ v_{i+1}）时线性插值求交点回合
    e* = e_i + (θ−v_i)·(e_{i+1}−e_i)/(v_{i+1}−v_i)；首点已过阈（v_0 ≥ θ）→
    τ = 首点回合（不外推到 0）。窗口语义：只认 τ ≤ window 的过阈（末段交点
    落在 window 外同样算未达）；窗口末仍未过阈 → 删失 (window, censored=True)。
    episodes 须升序（曲线按构造顺序追加，天然满足）。
    """
    if not episodes or len(episodes) != len(values):
        raise ValueError("episodes/values 须等长非空")
    if window <= 0:
        raise ValueError("window 须为正")
    if values[0] >= theta:
        return float(episodes[0]), False
    for (e_i, v_i), (e_j, v_j) in pairwise(zip(episodes, values)):
        if e_i > window:
            break
        if v_j < theta:
            continue
        crossing = e_i + (theta - v_i) * (e_j - e_i) / (v_j - v_i)
        if crossing <= window:
            return float(crossing), False
        break  # 首个跨越的交点已在窗口外，后续更晚 → 删失
    return float(window), True


def window_auc(episodes: Sequence[int], values: Sequence[float], window: int) -> float:
    """前 W 回合归一化 AUC = 梯形积分 ÷ W（计划 §5-2，值域随曲线量纲）。

    边界语义：取评估点 e ≤ W；W 落在两评估点之间时线性插值补 (W, v*) 右端点；
    曲线止于 W 之前（末点 e_last < W）时按末值延拓（LOCF）到 W；首评估点之前
    无观测、不外推（各臂评估调度相同，缺失前缀跨臂同质）。积分域 [e_first, W]，
    归一化分母恒为 W（"前 W 回合平均水平"口径）。
    """
    if not episodes or len(episodes) != len(values):
        raise ValueError("episodes/values 须等长非空")
    if window <= 0:
        raise ValueError("window 须为正")
    points = [(float(e), float(v)) for e, v in zip(episodes, values) if e <= window]
    if not points:
        raise ValueError("窗口内无评估点（首评估点已超过 W）")
    if len(points) < len(episodes):  # 存在 e > window 的下一点 → 插值出精确 W 端点
        e_next, v_next = float(episodes[len(points)]), float(values[len(points)])
        e_last, v_last = points[-1]
        points.append(
            (float(window), v_last + (v_next - v_last) * (window - e_last) / (e_next - e_last))
        )
    elif points[-1][0] < window:  # 曲线在 W 前结束 → LOCF
        points.append((float(window), points[-1][1]))
    integral = sum(
        (x2 - x1) * (y1 + y2) / 2.0 for (x1, y1), (x2, y2) in pairwise(points)
    )
    return integral / window


def mean_t_ci(values: Sequence[float]) -> tuple[float, float]:
    """样本均值 ±95%CI 半宽（t 分布，ddof=1，n=len(values) ≥ 2）。"""
    n = len(values)
    if n < 2:
        raise ValueError("CI 需 n>=2（n=1 无方差）")
    half = float(stats.t.ppf(0.5 + _CI_LEVEL / 2.0, n - 1)) * stdev(values) / sqrt(n)
    return mean(values), half


def asymptote(values: Sequence[float], k: int, *, strict: bool = True) -> tuple[float, float]:
    """渐近性能（计划 §5-3）：末 k 个评估点均值 ±95%CI 半宽（t 分布，n=k_eff）。

    k ≥ 2 契约：k_eff=1 无方差、CI 无定义 → ValueError（预注册 k=5，勿传 1）。
    strict=True（缺省）：点数 < k 抛 ValueError（预注册口径，防静默降级）；
    strict=False：k_eff = min(k, len(values))——短曲线降级可算（点数少 → CI
    自然变宽），聚合端（derive_theta/compute_table）对试点/短曲线用此口径。
    """
    k_eff = min(k, len(values)) if not strict else k
    if k_eff < 2:
        raise ValueError("渐近 CI 需 k>=2（k_eff=1 单点无 CI 可言；预注册 k=5，勿传 1）")
    if len(values) < k_eff:
        raise ValueError(f"评估点数 {len(values)} < k_eff={k_eff}")
    return mean_t_ci(values[-k_eff:])


def pooled_within_sd(groups: Mapping[str, Sequence[float]]) -> float:
    """合并组内标准差（θ 推导用，计划 §5-1）：sqrt(Σ_g Σ_i (x−mean_g)² / (N−G))。

    groups：组名 → 组内值列表（每组至少 1 值）；自由度 N−G ≥ 1 否则 ValueError。
    """
    total = sum(len(vals) for vals in groups.values())
    if not groups or total - len(groups) < 1:
        raise ValueError("合并 SD 需非空分组且总自由度 N−G ≥ 1")
    total_ss = sum((x - mean(vals)) ** 2 for vals in groups.values() for x in vals)
    return sqrt(total_ss / (total - len(groups)))


def paired_tests(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """逐种子配对检验（计划 §5 统计）：配对 t + Wilcoxon + 效应量。

    返回 {"t_p", "wilcoxon_p", "mean_diff", "cohens_d"}；mean_diff = mean(x−y)、
    cohens_d = mean_diff / sd(x−y)（配对 dz）。全零差（两臂逐种子同分）契约化
    输出 p=1、d=0（Wilcoxon 对零差序列无定义）；差恒非零且零方差时 d=±inf。
    需 n ≥ 2。
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("配对检验需 x/y 等长且 n>=2")
    diffs = [float(a) - float(b) for a, b in zip(x, y)]
    mean_diff = mean(diffs)
    if all(d == 0.0 for d in diffs):
        return {"t_p": 1.0, "wilcoxon_p": 1.0, "mean_diff": 0.0, "cohens_d": 0.0}
    sd_d = stdev(diffs)
    cohens_d = copysign(inf, mean_diff) if sd_d == 0.0 else mean_diff / sd_d
    wilcoxon_res = stats.wilcoxon(x, y)
    assert isinstance(wilcoxon_res, _HasPValue)
    return {
        "t_p": float(stats.ttest_rel(x, y).pvalue),
        "wilcoxon_p": float(wilcoxon_res.pvalue),
        "mean_diff": mean_diff,
        "cohens_d": cohens_d,
    }


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni 步降族校正（计划 §5 统计；自实现）。

    第 i 顺序统计量（升序 rank=i，0-based）乘族宽 (m−i)，运行 max 保单调
    不减，截断 1.0；返回按原始输入顺序排列的校正 p 值。
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(pvalues[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted
