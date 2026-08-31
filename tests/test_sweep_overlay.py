"""seam 1（工程接缝 1）：grid 顶层 `config_overlay:` 深合并契约测试。

覆盖：net/ppo 子树深合并且保留兄弟键、overlay 优先于预算字段、无/空 overlay
时与旧口径（仅 pretrain/adapt episodes 覆盖）逐位一致（锁定默认路径零漂移）、
plan_runs 把叠加值传播进 RunTask.config（下游 tune.py seam 的取值入口）、
非映射 overlay 显式拒绝。纯函数路径 + tmp_path 注入 paths（同 test_lineage
模式，不触真实 runs/）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trl_sb3.common.config import load_config, merge
from trl_sb3.run.sweep import _merged_config, plan_runs

SCENARIO = {"topo": "Abilene.gml", "avgrate": 500}


def _base_cfg(root: Path) -> dict[str, Any]:
    return merge(load_config(), {"paths": {"runs_dir": str(root / "runs"), "ckpts_dir": str(root / "ckpts")}})


def _mini_grid(**extra: Any) -> dict[str, Any]:
    """小网格：1 场景、预训练/适应各 1 回合、A1 单臂（同 test_lineage 口径）。"""
    grid: dict[str, Any] = {
        "pretrain": {"seeds": [0], "episodes": 1},
        "arms": ["A1"],
        "scenarios": [SCENARIO],
        "seeds": [0],
        "adapt_episodes": 1,
    }
    grid.update(extra)
    return grid


def test_overlay_net_layers_replaces_list_keeps_siblings(tmp_path: Path) -> None:
    """Given overlay net.layers=[287,256,128]；When _merged_config；Then net.layers
    整体替换（列表不做逐元素合并）、net 兄弟键与其余顶层节全部保留。"""
    base = _base_cfg(tmp_path)
    cfg = _merged_config(_mini_grid(config_overlay={"net": {"layers": [287, 256, 128]}}), base)
    assert cfg["net"]["layers"] == [287, 256, 128]
    assert base["net"]["layers"] == [287, 128, 64]  # 基线默认，证明替换真实生效
    for key in ("activation", "init", "separate_pi_vf"):
        assert cfg["net"][key] == base["net"][key]
    expected_legacy = merge(base, {"pretrain": {"episodes": 1}, "adapt": {"episodes": 1}})
    for key in expected_legacy:
        if key != "net":
            assert cfg[key] == expected_legacy[key]


def test_overlay_ppo_subtree_merges_without_dropping_siblings(tmp_path: Path) -> None:
    """Given overlay ppo.learning_rate/ent_coef 两键；When _merged_config；Then
    两键生效、ppo 其余键（n_steps/gamma/batch_size/...）原样保留。"""
    cfg = _merged_config(
        _mini_grid(config_overlay={"ppo": {"learning_rate": 1.0e-4, "ent_coef": 0.01}}), _base_cfg(tmp_path)
    )
    assert cfg["ppo"]["learning_rate"] == 1.0e-4
    assert cfg["ppo"]["ent_coef"] == 0.01
    base = _base_cfg(tmp_path)
    for key in base["ppo"]:
        if key not in ("learning_rate", "ent_coef"):
            assert cfg["ppo"][key] == base["ppo"][key]


def test_overlay_applies_after_budget_field_overrides(tmp_path: Path) -> None:
    """Given 网格 pretrain.episodes=1 且 overlay pretrain.episodes=3；When
    _merged_config；Then 结果为 3（overlay 在预算字段之后合入 → 优先）。"""
    cfg = _merged_config(
        _mini_grid(config_overlay={"pretrain": {"episodes": 3}}),
        _base_cfg(tmp_path),
    )
    assert cfg["pretrain"]["episodes"] == 3


def test_grid_without_or_empty_overlay_matches_episodes_only_behavior(tmp_path: Path) -> None:
    """Given 无 config_overlay 与空 config_overlay 两种网格；When _merged_config；
    Then 均与旧口径（仅 pretrain/adapt episodes 覆盖）逐位相等——默认路径零漂移。"""
    expected = merge(_base_cfg(tmp_path), {"pretrain": {"episodes": 1}, "adapt": {"episodes": 1}})
    assert _merged_config(_mini_grid(), _base_cfg(tmp_path)) == expected
    assert _merged_config(_mini_grid(config_overlay={}), _base_cfg(tmp_path)) == expected


def test_plan_runs_propagates_overlay_into_every_task_config(tmp_path: Path) -> None:
    """Given overlay net.layers 网格；When plan_runs；Then 每个 RunTask.config 携带
    叠加值（下游 seam 2/tune.py 的取值入口），任务清单形状不受 overlay 影响。"""
    plain = plan_runs(_mini_grid(), _base_cfg(tmp_path))
    tasks = plan_runs(_mini_grid(config_overlay={"net": {"layers": [287, 256, 128]}}), _base_cfg(tmp_path))
    assert [t.arm for t in tasks] == [t.arm for t in plain] == ["pretrain", "A1"]
    assert all(t.config["net"]["layers"] == [287, 256, 128] for t in tasks)


def test_non_mapping_overlay_raises_value_error(tmp_path: Path) -> None:
    """Given config_overlay 为列表（YAML 误写）；When _merged_config；Then
    ValueError 显式拒绝而非静默忽略（实验完整性：坏网格不得默默跑错超参）。"""
    with pytest.raises(ValueError, match="config_overlay"):
        _merged_config(_mini_grid(config_overlay=["net", "ppo"]), _base_cfg(tmp_path))
