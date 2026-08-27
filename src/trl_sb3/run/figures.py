"""论文图形层（M4，计划 §6 / checkbox #12）：归一化 + 图 1/图 2 纯构建函数。

归一化口径（论文同式，mdpi-submission-entropy/mdpi-article.tex:475 原文
"normalized to [0,1] using the maximum values observed across the
corresponding experimental runs"）：同一 (topo, rate) 场景内跨**全部参与 run**
（曲线点与 final 一并）取 r_mean_mean 最大值做分母，曲线、参考线、渐近柱共用。
场景标签沿用论文拼写 **NSFCNET**（文件名 NFSCNET.gml，同一网络）。

图 1 build_adaptation_figure：2×4 小倍数逐臂，种子均值 ±95%CI 带、τ 竖虚线
（全删失 → 画删失窗口 W 边界并标注 censored）、OSPF/ECMP/LB/RR 归一化水平
参考线；多场景时同子图内逐场景多曲线。
图 2 build_asymptote_figure：逐场景子图，各臂归一化渐近均值柱 + CI 误差棒，
OSPF/ECMP/LB/RR/A0 参照柱（final_mean，无 CI）并列；渐近不可估（None，
曲线 <2 点）的臂跳过并在图底注明。

本模块无 IO（输入 records/table，输出 Figure）；matplotlib 用 Agg 后端。
IO 编排与统计见 run/make_figures.py。
"""

from __future__ import annotations

from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 须在 pyplot 之前：无显示环境出图
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from trl_sb3.eval.metrics import mean_t_ci
from trl_sb3.run.aggregate import NON_SCORING_ARMS

# 论文拼写映射：文件名 NFSCNET.gml → 图表标签 NSFCNET（计划 §6，同一网络）
TOPO_LABELS: dict[str, str] = {"NFSCNET.gml": "NSFCNET"}
_ARM_ORDER = ("A1", "A1b", "A2", "A3", "A3b", "A4", "A5", "A6")
_REF_ARMS = ("OSPF", "ECMP", "LB", "RR")
_REF_BARS = ("OSPF", "ECMP", "LB", "RR", "A0")
_ARM_COLOR, _REF_COLOR = "#4C72B0", "#8C8C8C"


def topo_label(topo: str) -> str:
    """场景标签：NFSCNET.gml → NSFCNET（论文拼写），其余去 .gml 后缀。"""
    return TOPO_LABELS.get(topo, topo.removesuffix(".gml"))


def _scene_of(record: dict[str, Any]) -> tuple[str, int]:
    return (record["topo"], int(record["rate"]))


def _scene_text(scene: tuple[str, int]) -> str:
    return f"{topo_label(scene[0])}@{scene[1]}"


def normalization_denominators(records: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    """逐 (topo, rate) 场景：跨全部参与 run 的 r_mean_mean 最大值（tex:475 口径）。

    曲线点与 final 一并参与取 max——分母是"该场景全部实验运行中见过的最大值"，
    图 1 曲线/参考线与图 2 渐近柱共用同分母。无任何值的场景不入字典。
    """
    maxima: dict[tuple[str, int], float] = {}
    for record in records:
        values = [float(point["r_mean_mean"]) for point in record["curve"]]
        final = record["final"].get("r_mean_mean")
        if final is not None:
            values.append(float(final))
        if not values:
            continue
        key = _scene_of(record)
        maxima[key] = max(maxima.get(key, values[0]), max(values))
    return maxima


def normalize_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], float]]:
    """records →（归一副本, 分母表）：curve/final 的 r_mean_mean 除以场景分母。

    纯函数不改输入；归一后各场景值域 [0,1] 且最大值=1。分母 ≤0（奖励全非正）
    → ValueError（宁可失败勿默算）。
    """
    denominators = normalization_denominators(records)
    nonpositive = {scene: value for scene, value in denominators.items() if value <= 0.0}
    if nonpositive:
        raise ValueError(f"归一化分母须为正（场景最大 r_mean_mean ≤0）：{nonpositive}")
    normalized: list[dict[str, Any]] = []
    for record in records:
        denominator = denominators.get(_scene_of(record))
        if denominator is None:  # 无曲线无 final 的空记录，无处可归一
            continue
        normalized.append(
            {
                **record,
                "curve": [
                    {**point, "r_mean_mean": float(point["r_mean_mean"]) / denominator}
                    for point in record["curve"]
                ],
                "final": {**record["final"], "r_mean_mean": float(record["final"]["r_mean_mean"]) / denominator},
            }
        )
    return normalized, denominators


def _arm_scene_points(
    records: list[dict[str, Any]], arm: str, scene: tuple[str, int]
) -> dict[int, list[float]]:
    """(arm, scene) 内逐 episode 聚合各种子曲线值 → {episode: [种子值]}。"""
    points: dict[int, list[float]] = {}
    for record in records:
        if record["arm"] != arm or _scene_of(record) != scene:
            continue
        for point in record["curve"]:
            points.setdefault(int(point["episode"]), []).append(float(point["r_mean_mean"]))
    return points


def _final_mean(records: list[dict[str, Any]], arm: str, scene: tuple[str, int]) -> float | None:
    """(arm, scene) 的 final r_mean_mean 种子均值；无记录 → None。"""
    finals = [
        float(record["final"]["r_mean_mean"])
        for record in records
        if record["arm"] == arm and _scene_of(record) == scene and record["final"].get("r_mean_mean") is not None
    ]
    return mean(finals) if finals else None


def _ordered_arms(records: list[dict[str, Any]]) -> list[str]:
    """学习臂出场序：ARM_ORDER 在前，未知臂字典序殿后。"""
    present = {
        record["arm"] for record in records if record["curve"] and record["arm"] not in NON_SCORING_ARMS
    }
    return [arm for arm in _ARM_ORDER if arm in present] + sorted(present - set(_ARM_ORDER))


def _censored_rows(rows: dict[tuple[str, str, int], dict[str, Any]], arm: str, scenes: list[tuple[str, int]]) -> bool:
    """该臂是否存在全删失场景行（τ 竖线落在 W 上需要子图标注）。"""
    return any(
        (rows.get((arm, *scene)) or {}).get("tau") and rows[arm, *scene]["tau"]["n_censored"] >= rows[arm, *scene]["n_seeds"]
        for scene in scenes
    )


def build_adaptation_figure(
    records_norm: list[dict[str, Any]], table: list[dict[str, Any]], window: int
) -> Figure:
    """图 1（纯函数）：2×4 逐臂归一化适应曲线，种子均值 ±95%CI 带。

    τ 取 compute_table 行（episode 轴原量纲，不归一）：全删失 → 删失窗口 W
    边界虚线 + 子图标注 censored；未删失 → τ 均值虚线。OSPF/ECMP/LB/RR 逐
    场景水平参考线（灰点线）。多场景 → 同子图逐场景多色曲线。
    """
    scenes = sorted({_scene_of(r) for r in records_norm if r["arm"] not in NON_SCORING_ARMS})
    arms = _ordered_arms(records_norm)
    colors = dict(zip(scenes, plt.rcParams["axes.prop_cycle"].by_key()["color"], strict=False))
    rows = {(row["arm"], row["topo"], row["rate"]): row for row in table}
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.5), sharey=True, constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        if index >= len(arms):
            ax.set_visible(False)
            continue
        arm = arms[index]
        xmax = 0.0
        for scene in scenes:
            points = _arm_scene_points(records_norm, arm, scene)
            if not points:
                continue
            episodes = sorted(points)
            means = [mean(points[e]) for e in episodes]
            halves = [mean_t_ci(points[e])[1] if len(points[e]) >= 2 else None for e in episodes]
            ax.plot(
                episodes, means, color=colors[scene], marker="o", markersize=3,
                linewidth=1.5, label=_scene_text(scene) if index == 0 else None,
            )
            ax.fill_between(
                episodes,
                [m - (h or 0.0) for m, h in zip(means, halves)],
                [m + (h or 0.0) for m, h in zip(means, halves)],
                color=colors[scene], alpha=0.18, linewidth=0,
            )
            xmax = max(xmax, episodes[-1])
            row = rows.get((arm, *scene))
            if row and row["tau"]:
                all_censored = row["tau"]["n_censored"] >= row["n_seeds"]
                tau_x = float(window if all_censored else row["tau"]["mean"])
                ax.axvline(tau_x, color=colors[scene], linestyle="--", linewidth=1.2, alpha=0.8)
                xmax = max(xmax, tau_x)
        for scene in scenes:
            for ref in _REF_ARMS:
                value = _final_mean(records_norm, ref, scene)
                if value is not None:
                    ax.axhline(value, color="0.45", linestyle=":", linewidth=1.0)
        ax.set_xlim(0, xmax * 1.03)
        ax.set_title(arm, fontsize=11)
        if _censored_rows(rows, arm, scenes):
            ax.text(0.98, 0.95, "τ censored", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes.flat[0].set_ylim(-0.05, 1.05)
    fig.supxlabel("Episode")
    fig.supylabel("Normalized r_mean (per-scene max across runs)")
    fig.suptitle("Adaptation curves by arm (seed mean ±95% CI)", fontsize=13)
    handles = [Line2D([0], [0], color=colors[s], linewidth=1.5, label=_scene_text(s)) for s in scenes]
    if len(scenes) == 1:
        handles += [
            Line2D([0], [0], color="0.45", linestyle=":", label=ref)
            for ref in _REF_ARMS
            if _final_mean(records_norm, ref, scenes[0]) is not None
        ]
    else:
        handles.append(Line2D([0], [0], color="0.45", linestyle=":", label="OSPF/ECMP/LB/RR"))
    handles.append(Line2D([0], [0], color="0.15", linestyle="--", label=f"τ (at W={window} if censored)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 9), frameon=False)
    return fig


def _bar_order(arm: str) -> tuple[int, int | str]:
    """图 2 柱序：学习臂（ARM_ORDER 序）→ 参照（_REF_BARS 序）→ 未知殿后。"""
    if arm in _ARM_ORDER:
        return (0, _ARM_ORDER.index(arm))
    if arm in _REF_BARS:
        return (1, _REF_BARS.index(arm))
    return (2, arm)


def build_asymptote_figure(table: list[dict[str, Any]], denominators: dict[tuple[str, int], float]) -> Figure:
    """图 2（纯函数）：逐场景归一化渐近性能柱（±95%CI 误差棒）+ 参照柱并列。

    学习臂柱高 = asymptote.mean ÷ 场景分母、误差棒 = CI 半宽 ÷ 分母；
    参照（OSPF/ECMP/LB/RR/A0）= final_mean ÷ 分母（eval-only 无 CI）。
    渐近不可估（None，曲线 <2 点）的学习臂跳过并在图底注明。
    """
    scenes = sorted({(row["topo"], row["rate"]) for row in table})
    if not scenes:
        raise ValueError("指标表为空：无学习臂/基线行可画")
    ncols = min(len(scenes), 2)
    nrows = -(-len(scenes) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), squeeze=False, constrained_layout=True
    )
    for ax in axes.flat[len(scenes):]:
        ax.set_visible(False)
    skipped: list[str] = []
    for ax, scene in zip(axes.flat, scenes):
        denominator = denominators[scene]
        labels: list[str] = []
        heights: list[float] = []
        errors: list[float] = []
        bar_colors: list[str] = []
        for row in sorted((r for r in table if (r["topo"], r["rate"]) == scene), key=lambda r: _bar_order(r["arm"])):
            is_ref = row["arm"] in _REF_BARS
            if not is_ref and row["asymptote"] is None:
                skipped.append(f"{row['arm']}@{_scene_text(scene)}")
                continue
            labels.append(row["arm"])
            heights.append((row["asymptote"]["mean"] if row["asymptote"] else row["final_mean"]) / denominator)
            ci = row["asymptote"]["ci95"] if row["asymptote"] else None
            errors.append((ci[1] - ci[0]) / 2.0 / denominator if ci else 0.0)
            bar_colors.append(_REF_COLOR if is_ref else _ARM_COLOR)
        positions = range(len(labels))
        ax.bar(positions, heights, yerr=errors, capsize=3, color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(list(positions), labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(_scene_text(scene), fontsize=11)
        ax.set_ylabel("Normalized asymptotic performance" if ncols == 1 else None)
    fig.suptitle("Asymptotic performance by arm (normalized, ±95% CI)", fontsize=13)
    if skipped:
        fig.text(0.5, -0.02, f"Asymptote not estimable (curves <2 points): {', '.join(skipped)}",
                 ha="center", fontsize=8)
    return fig
