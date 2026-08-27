"""M3 聚合器契约测试：合成 runs 目录（按产物契约手写落盘）→ 记录/θ 推导/
指标表/配对差。不跑训练——manifest+eval.json+DONE 由测试直接写 tmp_path。

锁定：collect_runs 只扫 DONE、derive_theta 公式 max(OSPF, best−pooledσ) 与
prereg 定值优先、compute_table 行结构（学习臂三元组 / eval-only 仅 final /
pretrain 整行排除）、paired_arm_diffs 逐种子配对与共享种子守卫、holm_family。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trl_sb3.run.aggregate import (
    collect_runs,
    compute_table,
    derive_theta,
    holm_family,
    paired_arm_diffs,
)

TOPO = "Abilene.gml"
PREREG: dict[str, Any] = {
    "theta": None,
    "window_episodes": 500,
    "asymptote_k": 5,
    "alpha": 0.05,
    "primary_metric": "tau",
}
# 线性曲线（episode=100i）：A1 seed0 = 0.0..1.0、seed1 平移 +0.05；A2 减半。
_CURVE_A1_S0 = [{"episode": 100 * i, "r_mean_mean": 0.2 * (i - 1)} for i in range(1, 7)]
_CURVE_A1_S1 = [{"episode": 100 * i, "r_mean_mean": 0.2 * (i - 1) + 0.05} for i in range(1, 7)]
_CURVE_A2_S0 = [{"episode": 100 * i, "r_mean_mean": 0.1 * (i - 1)} for i in range(1, 7)]
_CURVE_A2_S1 = [{"episode": 100 * i, "r_mean_mean": 0.1 * (i - 1) + 0.05} for i in range(1, 7)]
_RECORD_KEYS = {
    "run_id", "arm", "topo", "rate", "seed", "episodes",
    "factors", "source_run_id", "curve", "final",
}


def _write_run(
    runs: Path,
    run_id: str,
    arm: str,
    seed: int,
    *,
    curve: list[dict[str, float]] | None = None,
    final: float = 0.1,
    done: bool = True,
) -> None:
    directory = runs / run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "arm": arm,
        "topo": TOPO,
        "rate": 500,
        "seed": seed,
        "episodes": 600,
        "factors": {"pretrain": False, "freeze": False, "pbrs": False},
        "source_run_id": None,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "eval.json").write_text(
        json.dumps({"curve": curve or [], "final": {"r_mean_mean": final}}), encoding="utf-8"
    )
    if done:
        (directory / "DONE").write_text("", encoding="utf-8")


def _pilot_records(tmp_path: Path) -> list[dict[str, Any]]:
    """合成试点盘：OSPF/ECMP 基线 + A1/A2 各 2 种子曲线 + 1 个无 DONE 行。"""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, "ospf-1", "OSPF", 0, final=0.10)
    _write_run(runs, "ecmp-1", "ECMP", 0, final=0.05)
    _write_run(runs, "a1-s0", "A1", 0, curve=_CURVE_A1_S0, final=0.60)
    _write_run(runs, "a1-s1", "A1", 1, curve=_CURVE_A1_S1, final=0.66)
    _write_run(runs, "a2-s0", "A2", 0, curve=_CURVE_A2_S0, final=0.30)
    _write_run(runs, "a2-s1", "A2", 1, curve=_CURVE_A2_S1, final=0.35)
    _write_run(runs, "a1-incomplete", "A1", 2, curve=_CURVE_A1_S0, done=False)
    return collect_runs(runs)


def test_collect_runs_reads_only_done_contract_fields(tmp_path: Path) -> None:
    """Given 6 个 DONE run + 1 个无 DONE；When collect_runs；Then 6 条记录、
    键=契约字段、无 DONE 行不入场。"""
    records = _pilot_records(tmp_path)
    assert len(records) == 6
    assert set(records[0]) == _RECORD_KEYS
    assert {r["run_id"] for r in records} == {"ospf-1", "ecmp-1", "a1-s0", "a1-s1", "a2-s0", "a2-s1"}


def test_derive_theta_ospf_best_arm_and_pooled_sd(tmp_path: Path) -> None:
    """Given A1 渐近均值 0.625（最优）、A2 0.325、种子级偏离 ±0.025、OSPF=0.10；
    When derive_theta；Then θ = max(0.10, 0.625 − sqrt(0.0025/2))。"""
    sigma = (0.0025 / 2) ** 0.5
    assert derive_theta(_pilot_records(tmp_path), PREREG) == pytest.approx(max(0.10, 0.625 - sigma))


def test_derive_theta_prereg_value_wins(tmp_path: Path) -> None:
    """Given prereg.theta=0.5（已预注册定值）；When derive_theta；Then 直接 0.5。"""
    assert derive_theta(_pilot_records(tmp_path), {**PREREG, "theta": 0.5}) == 0.5


def test_derive_theta_requires_ospf_and_learning_curves(tmp_path: Path) -> None:
    """Given 只有 OSPF 行（无学习臂曲线）；When derive_theta；Then ValueError。"""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, "ospf-1", "OSPF", 0, final=0.1)
    with pytest.raises(ValueError, match="OSPF"):
        derive_theta(collect_runs(runs), PREREG)


def test_compute_table_rows_censoring_and_eval_only(tmp_path: Path) -> None:
    """Given 试点样记录；When compute_table；Then A1 行 τ 插值未删失、W-AUC
    已知值、渐近均值 0.625；A2 行全删失（τ=W）；OSPF/ECMP 行仅 final_mean。"""
    records = _pilot_records(tmp_path)
    rows = {row["arm"]: row for row in compute_table(records, PREREG)}
    theta = derive_theta(records, PREREG)
    a1 = rows["A1"]
    assert a1["theta"] == pytest.approx(theta)
    assert a1["tau"]["n_censored"] == 0
    assert set(a1["tau"]["by_seed"]) == {0, 1}
    assert a1["window_auc"]["by_seed"] == {0: pytest.approx(0.32), 1: pytest.approx(0.36)}
    assert a1["asymptote"]["mean"] == pytest.approx(0.625)
    assert a1["asymptote"]["ci95"][0] < 0.625 < a1["asymptote"]["ci95"][1]
    a2 = rows["A2"]
    assert a2["tau"]["n_censored"] == 2
    assert a2["tau"]["by_seed"] == {0: 500.0, 1: 500.0}
    assert rows["OSPF"]["tau"] is None and rows["OSPF"]["final_mean"] == pytest.approx(0.10)
    assert rows["ECMP"]["final_mean"] == pytest.approx(0.05)


def test_compute_table_excludes_pretrain_rows(tmp_path: Path) -> None:
    """Given 混入源域 pretrain 行（同场景、渐近 2.0 远超各臂）；When
    compute_table + derive_theta；Then 表无 pretrain 行、θ 不受其影响。"""
    records = _pilot_records(tmp_path)
    runs = tmp_path / "runs"
    _write_run(
        runs, "pre-s0", "pretrain", 0,
        curve=[{"episode": 100 * i, "r_mean_mean": 2.0} for i in range(1, 7)], final=2.0,
    )
    all_records = collect_runs(runs)
    assert "pretrain" not in {row["arm"] for row in compute_table(all_records, PREREG)}
    assert derive_theta(all_records, PREREG) == pytest.approx(derive_theta(records, PREREG))


def test_paired_arm_diffs_pairs_by_seed(tmp_path: Path) -> None:
    """Given A1 finals {0:0.60, 1:0.66}、A2 {0:0.30, 1:0.35}；When 配对差；
    Then seeds=[0,1]、diffs=[0.30,0.31]、mean_diff=0.305、统计键齐。"""
    result = paired_arm_diffs(_pilot_records(tmp_path), "A1", "A2", TOPO, 500)
    assert result["seeds"] == [0, 1]
    assert result["diffs"] == pytest.approx([0.30, 0.31])
    assert result["mean_diff"] == pytest.approx(0.305)
    assert {"t_p", "wilcoxon_p", "cohens_d"} <= set(result)


def test_paired_arm_diffs_needs_shared_seeds(tmp_path: Path) -> None:
    """Given OSPF 行仅 seed 0；When A1 vs OSPF 配对；Then ValueError（共享种子<2）。"""
    with pytest.raises(ValueError, match="共享种子"):
        paired_arm_diffs(_pilot_records(tmp_path), "A1", "OSPF", TOPO, 500)


def test_holm_family_corrects_across_pairs() -> None:
    """Given 两对比较 t_p=[0.01,0.04]；When holm_family；Then Holm → [0.02,0.04]。"""
    assert holm_family([{"t_p": 0.01}, {"t_p": 0.04}]) == pytest.approx([0.02, 0.04])


def test_compute_table_survives_two_point_curves(tmp_path: Path) -> None:
    """Given 冒烟盘：曲线各仅 2 评估点（<k=5）+ OSPF 行；When compute_table；
    Then 不崩——渐近降级 k_eff=2、A1 行 tau 存在（θ 退化为 max(OSPF, best−σ)
    仍可算）；全 1 点曲线的臂渐近键为 None、θ 退化 OSPF 口径。"""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, "ospf-1", "OSPF", 0, final=0.10)
    _write_run(runs, "a1-s0", "A1", 0, curve=[{"episode": 100, "r_mean_mean": 0.1}, {"episode": 200, "r_mean_mean": 0.5}], final=0.50)
    _write_run(runs, "a2-s0", "A2", 0, curve=[{"episode": 100, "r_mean_mean": 0.2}], final=0.20)
    records = collect_runs(runs)
    rows = {row["arm"]: row for row in compute_table(records, PREREG)}
    assert rows["A1"]["tau"] is not None and set(rows["A1"]["tau"]["by_seed"]) == {0}
    assert rows["A1"]["asymptote"] is not None
    assert rows["A1"]["asymptote"]["mean"] == pytest.approx(0.3)
    assert rows["A2"]["asymptote"] is None  # 1 点曲线无 CI 可言
    # 唯一可估臂单种子 → pooled σ 无自由度取 0，θ = max(0.10, 0.3−0) = 0.3
    assert derive_theta(records, PREREG) == pytest.approx(0.3)
