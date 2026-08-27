"""M3 指标函数单测（计划 §5 三元组）：合成曲线已知值锁定。

覆盖：τ 线性插值（已知交点）/首点过阈/窗口删失/窗口边界跨界、W-AUC 已知
梯形值/窗口边界插值/LOCF 延拓、渐近 CI（k≥2 契约 + 独立公式复算）、
pooled σ 已知值、paired_tests 对称性与全零差契约、Holm 已知序列与封顶。
"""

from __future__ import annotations

import pytest
from scipy import stats as sps

from trl_sb3.eval.metrics import (
    asymptote,
    holm,
    mean_t_ci,
    paired_tests,
    pooled_within_sd,
    time_to_threshold,
    window_auc,
)


def test_time_to_threshold_interpolates_known_crossing() -> None:
    """Given 线性曲线 (100,0.1),(200,0.3),(300,0.5)、θ=0.2、W=500；When τ；
    Then 0.1→0.3 段交点 e=150，未删失。"""
    assert time_to_threshold([100, 200, 300], [0.1, 0.3, 0.5], 0.2, 500) == (150.0, False)


def test_time_to_threshold_first_point_already_over() -> None:
    """Given 首点 v0 ≥ θ；When τ；Then τ=首点回合（不外推到 0）。"""
    assert time_to_threshold([100, 200], [0.5, 0.6], 0.5, 500) == (100.0, False)


def test_time_to_threshold_censored_when_never_crosses() -> None:
    """Given 窗口内始终低于 θ；When τ；Then (W, True) 删失。"""
    assert time_to_threshold([100, 200, 300], [0.0, 0.1, 0.2], 0.5, 500) == (500.0, True)


def test_time_to_threshold_crossing_at_window_boundary() -> None:
    """Given (100,0.1),(300,0.5)：θ=0.3 交点恰在 e=200；When W=200；
    Then τ=200 未删失（≤W 计入）；θ=0.4 时交点 250>W → 删失。"""
    assert time_to_threshold([100, 300], [0.1, 0.5], 0.3, 200) == (200.0, False)
    assert time_to_threshold([100, 300], [0.1, 0.5], 0.4, 200) == (200.0, True)


def test_window_auc_known_trapezoid() -> None:
    """Given 线性曲线 [100..500]×[0,0.2,…,0.8]、W=500；When AUC；
    Then 梯形积分=160（首点前不外推）、÷500=0.32。"""
    assert window_auc([100, 200, 300, 400, 500], [0.0, 0.2, 0.4, 0.6, 0.8], 500) == pytest.approx(0.32)


def test_window_auc_interpolates_window_boundary() -> None:
    """Given (100,0),(300,1)、W=200 → 插值端点 (200,0.5)；When AUC；
    Then 积分=(0+0.5)/2×100=25、÷200=0.125。"""
    assert window_auc([100, 300], [0.0, 1.0], 200) == pytest.approx(0.125)


def test_window_auc_locf_when_curve_ends_before_window() -> None:
    """Given 曲线止于 200（值 0.5）、W=500；When AUC；Then LOCF 延拓：
    0.5×400/500=0.4。"""
    assert window_auc([100, 200], [0.5, 0.5], 500) == pytest.approx(0.4)


def test_asymptote_mean_and_t_ci() -> None:
    """Given 末两点 [4,5]、k=2；When 渐近；Then mean=4.5、半宽=t_{0.975,1}×
    sd/√n（独立公式复算：sd=√0.5、se=0.5）。"""
    center, half = asymptote([1.0, 2.0, 3.0, 4.0, 5.0], 2)
    expected_half = float(sps.t.ppf(0.975, 1)) * 0.5
    assert center == 4.5
    assert half == pytest.approx(expected_half)


def test_asymptote_and_mean_ci_require_min_two_points() -> None:
    """Given k=1（CI 无定义）、k>len、或 mean_t_ci 单值；When 调用；
    Then ValueError（契约 k≥2 / n≥2）。"""
    with pytest.raises(ValueError, match="k>=2"):
        asymptote([1.0, 2.0], 1)
    with pytest.raises(ValueError):
        asymptote([1.0, 2.0], 3)
    with pytest.raises(ValueError):
        mean_t_ci([1.0])


def test_asymptote_non_strict_degrades_k_to_available_points() -> None:
    """Given 2 点曲线 [3.0, 5.0]、k=5、strict=False；When 渐近；Then 降级
    k_eff=2 不抛：mean=4.0、半宽=t_{0.975,1}×sd(2)/√2（n=2 公式复算）；
    单点曲线即便 strict=False 仍 ValueError（单点无 CI 可言）。"""
    center, half = asymptote([3.0, 5.0], 5, strict=False)
    assert center == 4.0
    assert half == pytest.approx(float(sps.t.ppf(0.975, 1)) * 1.4142135623730951 / 2.0**0.5)
    with pytest.raises(ValueError, match="k>=2"):
        asymptote([4.0], 5, strict=False)


def test_pooled_within_sd_known_value() -> None:
    """Given 两组 [1,2,3]/[3,4,5]（各 SS=2）；When 合并 SD；Then
    sqrt((2+2)/(6−2))=1.0；自由度不足（单组单值）→ ValueError。"""
    assert pooled_within_sd({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]}) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pooled_within_sd({"a": [1.0]})


def test_paired_tests_symmetry_and_known_diff() -> None:
    """Given n=6 配对样本（差 [1,1,-1,2,1,1]）；When 正向 vs 逆向 paired_tests；
    Then mean_diff=5/6 且符号翻转、两 p 值不变、cohens_d 符号翻转。"""
    x = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    y = [9.0, 11.0, 15.0, 14.0, 17.0, 19.0]
    forward = paired_tests(x, y)
    backward = paired_tests(y, x)
    assert forward["mean_diff"] == pytest.approx(5.0 / 6.0)
    assert backward["mean_diff"] == pytest.approx(-5.0 / 6.0)
    assert forward["t_p"] == pytest.approx(backward["t_p"])
    assert forward["wilcoxon_p"] == pytest.approx(backward["wilcoxon_p"])
    assert forward["cohens_d"] == pytest.approx(-backward["cohens_d"])


def test_paired_tests_all_zero_diffs_defined_as_no_effect() -> None:
    """Given 两臂逐种子同分；When paired_tests；Then 契约化输出 p=1、d=0
    （Wilcoxon 对零差序列无定义）。"""
    result = paired_tests([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result == {"t_p": 1.0, "wilcoxon_p": 1.0, "mean_diff": 0.0, "cohens_d": 0.0}


def test_holm_known_sequence() -> None:
    """Given 经典序列 [0.01,0.04,0.03,0.005]；When Holm；Then 顺序统计量乘
    (4,3,2,1) + 运行 max → 原序 [0.03,0.06,0.06,0.02]。"""
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_empty_family() -> None:
    """Given 大 p 族 [0.9,0.95] 与空族；When Holm；Then 封顶 1.0；空族 → []."""
    assert holm([0.9, 0.95]) == pytest.approx([1.0, 1.0])
    assert holm([]) == []
