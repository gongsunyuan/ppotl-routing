"""指标聚合器（M3，计划 §5/§6）：runs 目录 → 逐 run 记录 → 指标表与配对比较。

读盘只经产物契约字段：manifest（arm/topo/rate/seed/factors/source_run_id/
episodes）+ eval.json（curve/final）+ DONE（唯一完整性判据；无 DONE 的目录
含 FAILED 跳过）。指标与统计实现全在 eval/metrics（纯函数）；本模块只做
IO + 分组编排。无 CLI——M4 make_figures 复用本模块。
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from trl_sb3.eval.metrics import (
    asymptote,
    holm,
    mean_t_ci,
    paired_tests,
    pooled_within_sd,
    time_to_threshold,
    window_auc,
)

# 不进 θ 推导的 arm：eval-only 基线（OSPF/启发式/A0，curve=[]）+ 源域预训练行
# （非对比臂，损失曲线材料由 orchestrator 直读其 eval.json，不走指标表）。
NON_SCORING_ARMS = frozenset({"OSPF", "ECMP", "LB", "RR", "A0", "pretrain"})


def collect_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """扫描 runs 目录：每个含 DONE 的 run 目录 → manifest+eval.json 摘要记录。

    记录键：run_id/arm/topo/rate/seed/episodes/factors/source_run_id/curve/final。
    无 DONE 的目录（未完成或 FAILED）跳过——契约规定 DONE 最后写，缺 DONE 即
    不完整产物，聚合端不消费。
    """
    records: list[dict[str, Any]] = []
    for directory in sorted(Path(runs_dir).iterdir()):
        if not directory.is_dir() or not (directory / "DONE").exists():
            continue
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        eval_data = json.loads((directory / "eval.json").read_text(encoding="utf-8"))
        records.append(
            {
                "run_id": manifest["run_id"],
                "arm": manifest["arm"],
                "topo": manifest["topo"],
                "rate": int(manifest["rate"]),
                "seed": int(manifest["seed"]),
                "episodes": int(manifest["episodes"]),
                "factors": manifest.get("factors"),
                "source_run_id": manifest.get("source_run_id"),
                "curve": eval_data.get("curve", []),
                "final": eval_data.get("final", {}),
            }
        )
    return records


def _curve_series(record: dict[str, Any]) -> tuple[list[int], list[float]]:
    """eval.json curve → (episodes, r_mean_mean 值序列)（曲线主指标）。"""
    return (
        [int(point["episode"]) for point in record["curve"]],
        [float(point["r_mean_mean"]) for point in record["curve"]],
    )


def derive_theta(records: list[dict[str, Any]], prereg: dict[str, Any]) -> float:
    """θ 推导（计划 §5-1）：θ = max(OSPF 贪心分, 最优臂渐近 − 合并组内 σ)。

    prereg["theta"] 非 None 时直接返回（预注册定值优先，防事后挑 θ）。records
    为同一 (topo, rate) 场景的记录：最优臂渐近 = 各学习臂（curve 非空且非
    NON_SCORING_ARMS）种子渐近均值的最大者（渐近 = 末 prereg["asymptote_k"]
    点均值）；合并 σ = 学习臂间 pooled-within-SD（种子级渐近值按臂分组）。
    缺 OSPF 行或无学习臂记录 → ValueError（数据不全宁可失败，勿默算）；
    学习臂曲线全 <2 点（无法估渐近）→ θ 退化 OSPF 口径（max(OSPF)）。
    """
    if prereg.get("theta") is not None:
        return float(prereg["theta"])
    ospf_scores = [r["final"]["r_mean_mean"] for r in records if r["arm"] == "OSPF"]
    k = int(prereg["asymptote_k"])
    # strict=False：试点/短曲线降级——点数不足 k 时渐近仍可算（CI 宽度自然
    # 变大）；曲线 <2 点的记录无法估渐近，不入分组。
    learners = [r for r in records if r["curve"] and r["arm"] not in NON_SCORING_ARMS]
    if not ospf_scores or not learners:
        raise ValueError("θ 推导需同场景 OSPF 行与至少一个学习臂曲线")
    arm_asymptotes: dict[str, list[float]] = {}
    for record in learners:
        _, values = _curve_series(record)
        if len(values) >= 2:
            arm_asymptotes.setdefault(record["arm"], []).append(asymptote(values, k, strict=False)[0])
    if not arm_asymptotes:  # 学习臂曲线全 <2 点 → 无渐近可估，退化 OSPF 口径
        return max(ospf_scores)
    best = max(mean(vals) for vals in arm_asymptotes.values())
    # pooled σ 需自由度 N−G ≥ 1（镜像 pooled_within_sd 契约）；不足（如单臂单种子）
    # → σ=0，θ = max(OSPF, best) 继续可算
    n_points = sum(len(vals) for vals in arm_asymptotes.values())
    sigma = pooled_within_sd(arm_asymptotes) if n_points > len(arm_asymptotes) else 0.0
    return max(max(ospf_scores), best - sigma)


def _summary(values: list[float]) -> dict[str, Any]:
    """种子间汇总：mean ±95%CI（t 分布；n<2 时 ci95=None）。"""
    if len(values) < 2:
        return {"mean": mean(values), "ci95": None}
    center, half = mean_t_ci(values)
    return {"mean": center, "ci95": [center - half, center + half]}


def compute_table(records: list[dict[str, Any]], prereg: dict[str, Any]) -> list[dict[str, Any]]:
    """逐 (arm, topo, rate) 组指标行（计划 §5 三元组 + final 均值）。

    学习臂（curve 非空）行：theta + tau{by_seed, n_censored, mean, ci95} +
    window_auc{by_seed, mean, ci95} + asymptote{by_seed:[mean, 半宽], mean, ci95}；
    eval-only 行（OSPF/ECMP/LB/RR/A0）：仅 final_mean（曲线指标 None）；源域
    pretrain 行整行排除。θ 逐 (topo, rate) 场景取 derive_theta（prereg.theta
    定值则直用）；场景无学习臂时不推导。行按 (arm, topo, rate) 字典序。
    """
    window = int(prereg["window_episodes"])
    k = int(prereg["asymptote_k"])
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["arm"], record["topo"], record["rate"]), []).append(record)
    thetas: dict[tuple[str, int], float | None] = {}
    for topo, rate in sorted({(r["topo"], r["rate"]) for r in records}):
        scene = [r for r in records if r["topo"] == topo and r["rate"] == rate]
        has_learner = any(r["curve"] and r["arm"] not in NON_SCORING_ARMS for r in scene)
        thetas[(topo, rate)] = derive_theta(scene, prereg) if has_learner else None
    rows: list[dict[str, Any]] = []
    for (arm, topo, rate), members in sorted(groups.items()):
        if arm == "pretrain":
            continue  # 源域预训练非对比臂
        members.sort(key=lambda r: r["seed"])
        row: dict[str, Any] = {
            "arm": arm,
            "topo": topo,
            "rate": rate,
            "n_seeds": len(members),
            "theta": None,
            "tau": None,
            "window_auc": None,
            "asymptote": None,
            "final_mean": mean([m["final"]["r_mean_mean"] for m in members]),
        }
        curves = [m for m in members if m["curve"]]
        if not curves:  # eval-only 行
            rows.append(row)
            continue
        tau_by_seed: dict[int, float] = {}
        censored = 0
        asym_by_seed: dict[int, list[float]] = {}
        theta = thetas[(topo, rate)]
        # 本组 curve 非空 ⇒ 场景 has_learner 为真 ⇒ theta 已推导（非 None）
        assert theta is not None
        for member in curves:
            episodes, values = _curve_series(member)
            tau, is_censored = time_to_threshold(episodes, values, theta, window)
            tau_by_seed[member["seed"]] = tau
            censored += int(is_censored)
            if len(values) >= 2:  # strict=False 短曲线降级；<2 点无 CI 可言，该种子不进渐近
                asym_by_seed[member["seed"]] = list(asymptote(values, k, strict=False))
        row["theta"] = theta
        row["tau"] = {"by_seed": tau_by_seed, "n_censored": censored, **_summary(list(tau_by_seed.values()))}
        auc_by_seed = {
            member["seed"]: window_auc(*_curve_series(member), window) for member in curves
        }
        row["window_auc"] = {"by_seed": auc_by_seed, **_summary(list(auc_by_seed.values()))}
        # 全部种子曲线 <2 点时无渐近可估 → 整键 None（下游勿假设恒有）
        row["asymptote"] = (
            {"by_seed": asym_by_seed, **_summary([pair[0] for pair in asym_by_seed.values()])}
            if asym_by_seed
            else None
        )
        rows.append(row)
    return rows


def paired_arm_diffs(
    records: list[dict[str, Any]], arm_a: str, arm_b: str, topo: str, rate: int
) -> dict[str, Any]:
    """逐种子配对差（final r_mean_mean，同 topo/rate 场景；计划 §5 统计）。

    返回 {arm_a, arm_b, topo, rate, seeds, diffs(a−b), t_p, wilcoxon_p,
    mean_diff, cohens_d}。Holm 校正不在单对内做——多臂对比族由调用方收集
    全部配对结果后 holm_family() 统一校正。共享种子 < 2 → ValueError。
    """
    def finals(arm: str) -> dict[int, float]:
        return {
            r["seed"]: r["final"]["r_mean_mean"]
            for r in records
            if r["arm"] == arm and r["topo"] == topo and r["rate"] == rate
        }

    scores_a, scores_b = finals(arm_a), finals(arm_b)
    seeds = sorted(set(scores_a) & set(scores_b))
    if len(seeds) < 2:
        raise ValueError(f"配对需 >=2 共享种子：{arm_a} vs {arm_b} @ {topo}/{rate} 只有 {seeds}")
    values_a = [scores_a[s] for s in seeds]
    values_b = [scores_b[s] for s in seeds]
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "topo": topo,
        "rate": rate,
        "seeds": seeds,
        "diffs": [a - b for a, b in zip(values_a, values_b)],
        **paired_tests(values_a, values_b),
    }


def holm_family(pair_results: list[dict[str, Any]]) -> list[float]:
    """配对族 Holm 校正：对 paired_arm_diffs 结果列表的主指标 p 值族（t_p）校正。"""
    return holm([result["t_p"] for result in pair_results])
