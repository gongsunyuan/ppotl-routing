"""M2-4 sweep 编排器契约测试（计划 §7-7 lineage、D5、§6 编排）。

覆盖：lineage 拒绝（缺源 ckpt 迁移臂不启动、--strict 抛错）、dry-run 全清单
（状态字符串、不落盘）、--filter 子串、DONE 幂等跳过（无重复训练）、单 run
崩溃隔离、计划 run_id 预计算与实际产物目录名逐位一致（lineage 根基，必须锁定）。
全部走 tmp paths 注入 config（同 test_runner_contract 模式）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import trl_sb3.run.sweep as sweep_module
from trl_sb3.common.config import load_config, merge
from trl_sb3.run.sweep import plan_runs, run_sweep, task_run_id

SCENARIO = {"topo": "Abilene.gml", "avgrate": 500}


def _base_cfg(root: Path) -> dict[str, Any]:
    return merge(load_config(), {"paths": {"runs_dir": str(root / "runs"), "ckpts_dir": str(root / "ckpts")}})


def _mini_grid(
    *,
    pretrain_seeds: list[int],
    arms: list[str],
    seeds: list[int] | None = None,
    zeroshot: bool = False,
    ospf: bool = False,
) -> dict[str, Any]:
    """小网格：1 场景（Abilene@500）、预训练/适应各 1 回合，产物最小化。"""
    return {
        "pretrain": {"seeds": pretrain_seeds, "episodes": 1},
        "arms": arms,
        "scenarios": [SCENARIO],
        "seeds": seeds if seeds is not None else [0],
        "adapt_episodes": 1,
        "zeroshot": zeroshot,
        "ospf": ospf,
    }


def _grid_yaml(root: Path, grid: dict[str, Any]) -> Path:
    path = root / "grid.yaml"
    path.write_text(yaml.safe_dump(grid, sort_keys=False), encoding="utf-8")
    return path


def _task_run_dir(cfg: dict[str, Any], grid: dict[str, Any], arm: str) -> Path:
    task = next(t for t in plan_runs(grid, cfg) if t.arm == arm)
    return Path(cfg["paths"]["runs_dir"], task_run_id(task))


def test_lineage_guard_rejects_transfer_arm_without_source(tmp_path: Path) -> None:
    """Given 网格无预训练（A2 源 ckpt 必缺）；When run_sweep；Then A2 SKIP-LINEAGE
    且不产生 run 目录、A1 照常 done；--strict 时抛 RuntimeError。"""
    cfg = _base_cfg(tmp_path)
    grid_dict = _mini_grid(pretrain_seeds=[], arms=["A1", "A2"])
    grid = _grid_yaml(tmp_path, grid_dict)
    summary = run_sweep(grid, base_config=cfg)
    assert (summary.planned, summary.done, summary.skipped_lineage, summary.failed) == (2, 1, 1, 0)
    assert not _task_run_dir(cfg, grid_dict, "A2").exists()
    assert (_task_run_dir(cfg, grid_dict, "A1") / "DONE").exists()
    with pytest.raises(RuntimeError, match="lineage"):
        run_sweep(grid, base_config=cfg, strict=True)


def test_dry_run_lists_all_planned_without_creating_dirs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Given 2 种子小网格（预训练 2 + A1/A2×2 种子 + A0×2 + OSPF×1 = 9 任务、全新盘）；
    When run_sweep(dry_run=True)；Then 9 行全 NEW、无 SKIP、无任何 run 目录、done=0。"""
    grid = _grid_yaml(tmp_path, _mini_grid(pretrain_seeds=[0, 1], arms=["A1", "A2"], seeds=[0, 1], zeroshot=True, ospf=True))
    summary = run_sweep(grid, dry_run=True, base_config=_base_cfg(tmp_path))
    assert (summary.planned, summary.done, summary.skipped_done, summary.skipped_lineage) == (9, 0, 0, 0)
    output = capsys.readouterr().out
    assert output.count("NEW") == 9
    assert "SKIP" not in output
    assert "arm=OSPF" in output and "arm=A0" in output
    assert not (tmp_path / "runs").exists()


def test_filter_substring_matches_arm_or_topo(tmp_path: Path) -> None:
    """Given 9 任务网格（预训练行 topo=CERNET.gml、其余 Abilene）；When dry_run +
    --filter；Then 子串命中 arm（A2→2）或 topo（Abilene→7，预训练行不命中）的
    任务保留，无命中（NFSCNET→0）计划为空。"""
    grid = _grid_yaml(tmp_path, _mini_grid(pretrain_seeds=[0, 1], arms=["A1", "A2"], seeds=[0, 1], zeroshot=True, ospf=True))
    cfg = _base_cfg(tmp_path)
    assert run_sweep(grid, dry_run=True, filters="A2", base_config=cfg).planned == 2
    assert run_sweep(grid, dry_run=True, filters="Abilene", base_config=cfg).planned == 7
    assert run_sweep(grid, dry_run=True, filters="NFSCNET", base_config=cfg).planned == 0


def test_done_skip_is_idempotent_rerun(tmp_path: Path) -> None:
    """Given 已完整跑过的小网格（预训练 0 + A1）；When 再次 run_sweep；Then 两任务
    全 SKIP-DONE、done=0、A1 metrics.csv 行数不变（无重复训练）。"""
    grid_dict = _mini_grid(pretrain_seeds=[0], arms=["A1"])
    grid = _grid_yaml(tmp_path, grid_dict)
    cfg = _base_cfg(tmp_path)
    first = run_sweep(grid, base_config=cfg)
    assert (first.planned, first.done, first.failed) == (2, 2, 0)
    metrics = _task_run_dir(cfg, grid_dict, "A1") / "metrics.csv"
    lines_before = len(metrics.read_text(encoding="utf-8").splitlines())
    second = run_sweep(grid, base_config=cfg)
    assert (second.planned, second.done, second.skipped_done, second.failed) == (2, 0, 2, 0)
    assert len(metrics.read_text(encoding="utf-8").splitlines()) == lines_before


def test_single_run_crash_does_not_kill_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given A2 的 run_arm 被注入异常；When run_sweep（预训练 0 + A1 + A2）；
    Then sweep 不抛、failed=1、其余 2 done、A2 无 DONE。"""
    real_run_arm = sweep_module.run_arm

    def _boom_arm(arm: str, **kwargs: Any) -> Path:
        if arm == "A2":
            raise RuntimeError("boom-A2")
        return real_run_arm(arm, **kwargs)

    monkeypatch.setattr(sweep_module, "run_arm", _boom_arm)
    grid_dict = _mini_grid(pretrain_seeds=[0], arms=["A1", "A2"])
    grid = _grid_yaml(tmp_path, grid_dict)
    cfg = _base_cfg(tmp_path)
    summary = run_sweep(grid, base_config=cfg)
    assert (summary.planned, summary.done, summary.failed) == (3, 2, 1)
    assert not (_task_run_dir(cfg, grid_dict, "A2") / "DONE").exists()


def test_planned_run_ids_match_produced_run_dirs(tmp_path: Path) -> None:
    """Given 全链小网格（预训练/A1/A2/A0/OSPF 各 1）；When run_sweep 实跑；Then 落盘
    run 目录名集合与 plan 预计算 run_id 集合逐位相等（源 id 预计算≠实际=lineage 失效），
    A2/A0 的 source_run_id == 预训练任务 run_id，计划顺序预训练在前。"""
    grid_dict = _mini_grid(pretrain_seeds=[0], arms=["A1", "A2"], zeroshot=True, ospf=True)
    grid = _grid_yaml(tmp_path, grid_dict)
    cfg = _base_cfg(tmp_path)
    tasks = plan_runs(grid_dict, cfg)
    assert [t.arm for t in tasks] == ["pretrain", "A1", "A2", "A0", "OSPF"]
    pretrain_id = task_run_id(tasks[0])
    assert tasks[2].source_run_id == pretrain_id and tasks[3].source_run_id == pretrain_id
    summary = run_sweep(grid, base_config=cfg)
    assert (summary.planned, summary.done, summary.failed) == (5, 5, 0)
    produced = {d.name for d in Path(cfg["paths"]["runs_dir"]).iterdir()}
    assert produced == {task_run_id(t) for t in tasks}


def test_heuristic_grid_key_plans_and_produces_rows(tmp_path: Path) -> None:
    """Given heuristic 键网格（ECMP/LB/RR + 预训练/A1/A0/OSPF，heuristic_episodes=1
    注入）；When run_sweep 实跑；Then 计划顺序启发式在 OSPF 后、7 任务全 done、
    落盘目录名==预计算 run_id 集合（含启发式行）；未知启发式名 plan 阶段 ValueError。"""
    cfg = merge(_base_cfg(tmp_path), {"eval": {"heuristic_episodes": 1}})
    grid_dict = {
        **_mini_grid(pretrain_seeds=[0], arms=["A1"], zeroshot=True, ospf=True),
        "heuristic": ["ECMP", "LB", "RR"],
    }
    grid = _grid_yaml(tmp_path, grid_dict)
    tasks = plan_runs(grid_dict, cfg)
    assert [t.arm for t in tasks] == ["pretrain", "A1", "A0", "OSPF", "ECMP", "LB", "RR"]
    assert [t.episodes for t in tasks[4:]] == [1, 1, 1]
    summary = run_sweep(grid, base_config=cfg)
    assert (summary.planned, summary.done, summary.failed) == (7, 7, 0)
    produced = {d.name for d in Path(cfg["paths"]["runs_dir"]).iterdir()}
    assert produced == {task_run_id(t) for t in tasks}
    with pytest.raises(ValueError, match="未知启发式"):
        plan_runs({**grid_dict, "heuristic": ["XX"]}, cfg)
