"""缝 5：sweep DONE-skip 的 manifest-episodes 一致性守卫（D1 OSPF 5→10 的系统性防污染）。

背景：eval_episodes 等协议/配置变更后，旧协议下评估出的 DONE run 目录 run_id 不变
（确定性 id 不含 episodes 因素）→ DONE-skip 会静默放过陈旧行污染聚合表。守卫契约：
SKIP-DONE（实跑与 dry-run 走同一分支，冻结门依赖 dry-run 也能触发）时读
runs/<run_id>/manifest.json，manifest["episodes"] 必须等于当前计划 RunTask.episodes
（eval/heuristic/训练预算的现值）；不一致 / manifest 缺失或损坏 → RuntimeError（报
run_id、两侧数字与补救）；一致 → 照常跳过；NEW 行不触发守卫（不读 manifest）。
全部 tmp paths 注入 config（同 test_lineage 模式）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

import trl_sb3.run.sweep as sweep_module
from trl_sb3.common.config import load_config, merge
from trl_sb3.common.logging_utils import write_manifest
from trl_sb3.common.run_artifacts import build_manifest, mark_done
from trl_sb3.run.sweep import RunTask, plan_runs, run_sweep, task_run_id

SCENARIO_TOPO: str = "Abilene.gml"
SCENARIO_RATE: int = 500
# 网格场景行（dict 形态，喂 plan_runs）；精确因子值用上面的单值常量传 build_manifest。
SCENARIO: dict[str, Any] = {"topo": SCENARIO_TOPO, "avgrate": SCENARIO_RATE}


def _base_cfg(root: Path, *, eval_episodes: int) -> dict[str, Any]:
    """tmp 根 + eval_episodes 注入（模拟 D1 的 5→10 翻转）。"""
    return merge(
        load_config(),
        {
            "paths": {"runs_dir": str(root / "runs"), "ckpts_dir": str(root / "ckpts")},
            "eval": {"eval_episodes": eval_episodes},
        },
    )


def _ospf_grid() -> dict[str, Any]:
    """最小 OSPF-only 网格：单场景单任务（OSPF 行 episodes=eval_episodes）。"""
    return {
        "pretrain": {"seeds": [], "episodes": 1},
        "arms": [],
        "scenarios": [SCENARIO],
        "seeds": [0],
        "adapt_episodes": 1,
        "zeroshot": False,
        "ospf": True,
    }


def _grid_yaml(root: Path, grid: dict[str, Any]) -> Path:
    path = root / "grid.yaml"
    path.write_text(yaml.safe_dump(grid, sort_keys=False), encoding="utf-8")
    return path


def _ospf_row(cfg: dict[str, Any], grid: dict[str, Any]) -> tuple[Path, int]:
    """(OSPF 行 run 目录, 计划 episodes)——run_id 用 plan_runs+task_run_id 预计算。"""
    (task,) = plan_runs(grid, cfg)
    return Path(cfg["paths"]["runs_dir"], task_run_id(task)), task.episodes


def _fake_row(run_dir: Path, cfg: dict[str, Any], *, episodes: int, done: bool = True) -> None:
    """伪造产物行：真实 build_manifest 口径（episodes 注入）写 manifest.json；done=True
    再补 DONE 标记（metrics.csv/eval.json 缺席不影响守卫与跳过语义）。"""
    write_manifest(
        run_dir / "manifest.json",
        build_manifest(
            run_dir.name,
            "OSPF",
            SCENARIO_TOPO,
            SCENARIO_RATE,
            0,
            factors={"pretrain": False, "freeze": False, "pbrs": False},
            source_run_id=None,
            episodes=episodes,
            device="cpu",
            config=cfg,
        ),
    )
    if done:
        mark_done(run_dir)


def _forbid_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    """_execute 注入 boom：被调即 AssertionError——SKIP-DONE 行不得触发任何执行。"""

    def _boom(task: RunTask, _device: str) -> Path:
        raise AssertionError(f"SKIP-DONE 行不得触发执行：arm={task.arm}")

    monkeypatch.setattr(sweep_module, "_execute", _boom)


def test_skip_done_stale_episodes_raises_in_dry_run(tmp_path: Path) -> None:
    """Given OSPF 行 DONE 且 manifest episodes=5、当前计划 eval_episodes=10（D1 翻转后）；
    When run_sweep(dry_run=True)（冻结门验证面）；Then RuntimeError 报 run_id 与两侧
    episodes 数字及补救（删目录重生成），dry-run 不得静默放过陈旧行。"""
    cfg = _base_cfg(tmp_path, eval_episodes=10)
    grid_dict = _ospf_grid()
    grid = _grid_yaml(tmp_path, grid_dict)
    run_dir, planned = _ospf_row(cfg, grid_dict)
    assert planned == 10
    _fake_row(run_dir, cfg, episodes=5)
    with pytest.raises(RuntimeError, match=re.escape(run_dir.name)) as excinfo:
        run_sweep(grid, dry_run=True, base_config=cfg)
    message = str(excinfo.value)
    assert "episodes=5" in message and "episodes=10" in message


def test_skip_done_stale_episodes_raises_in_real_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given 同上陈旧行；When run_sweep 实跑（resume 缺省 True，_execute 注入 boom）；
    Then 同一 SKIP-DONE 分支硬失败——实跑路径（status != "NEW" 进入共用分支）与
    dry-run（dry_run=True 进入）覆盖一致，且不触发任何执行。"""
    cfg = _base_cfg(tmp_path, eval_episodes=10)
    grid_dict = _ospf_grid()
    grid = _grid_yaml(tmp_path, grid_dict)
    run_dir, _ = _ospf_row(cfg, grid_dict)
    _fake_row(run_dir, cfg, episodes=5)
    _forbid_execute(monkeypatch)
    with pytest.raises(RuntimeError, match=re.escape(run_dir.name)):
        run_sweep(grid, base_config=cfg)


def test_skip_done_matching_episodes_skips_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given OSPF 行 DONE 且 manifest episodes=10 == 计划 eval_episodes=10；When
    run_sweep 实跑（_execute 注入 boom）；Then SKIP-DONE 计 1、done=0、无异常——
    一致行照常跳过，守卫零误伤。"""
    cfg = _base_cfg(tmp_path, eval_episodes=10)
    grid_dict = _ospf_grid()
    grid = _grid_yaml(tmp_path, grid_dict)
    run_dir, planned = _ospf_row(cfg, grid_dict)
    _fake_row(run_dir, cfg, episodes=planned)
    _forbid_execute(monkeypatch)
    summary = run_sweep(grid, base_config=cfg)
    assert (summary.planned, summary.skipped_done, summary.done, summary.failed) == (1, 1, 0, 0)


def test_skip_done_missing_manifest_raises(tmp_path: Path) -> None:
    """Given DONE 标记存在但 manifest.json 缺失；When run_sweep(dry_run=True)；Then
    RuntimeError 报 run_id 与补救——损坏/不完整产物不得被静默跳过。"""
    cfg = _base_cfg(tmp_path, eval_episodes=10)
    grid_dict = _ospf_grid()
    grid = _grid_yaml(tmp_path, grid_dict)
    run_dir, _ = _ospf_row(cfg, grid_dict)
    run_dir.mkdir(parents=True)
    mark_done(run_dir)
    with pytest.raises(RuntimeError, match=re.escape(run_dir.name)):
        run_sweep(grid, dry_run=True, base_config=cfg)


def test_skip_done_corrupt_manifest_raises(tmp_path: Path) -> None:
    """Given DONE + manifest.json 为非 JSON 垃圾文本；When run_sweep(dry_run=True)；
    Then 同缺失一样硬失败（损坏 != 可跳过）。"""
    cfg = _base_cfg(tmp_path, eval_episodes=10)
    grid_dict = _ospf_grid()
    grid = _grid_yaml(tmp_path, grid_dict)
    run_dir, _ = _ospf_row(cfg, grid_dict)
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    mark_done(run_dir)
    with pytest.raises(RuntimeError, match=re.escape(run_dir.name)):
        run_sweep(grid, dry_run=True, base_config=cfg)


def test_new_task_with_mismatched_manifest_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given run 目录有 manifest（episodes=999 必不一致）但无 DONE 标记；When
    run_sweep 实跑（_execute 打桩记录）；Then 任务按 NEW 正常执行（done=1）、无异常
    ——守卫只作用于 SKIP-DONE，NEW 行不读 manifest、零误伤。"""
    cfg = _base_cfg(tmp_path, eval_episodes=10)
    grid_dict = _ospf_grid()
    grid = _grid_yaml(tmp_path, grid_dict)
    run_dir, _ = _ospf_row(cfg, grid_dict)
    _fake_row(run_dir, cfg, episodes=999, done=False)
    executed: list[str] = []

    def _stub(task: RunTask, _device: str) -> Path:
        executed.append(task.arm)
        return run_dir

    monkeypatch.setattr(sweep_module, "_execute", _stub)
    summary = run_sweep(grid, base_config=cfg)
    assert (summary.planned, summary.done, summary.skipped_done, summary.failed) == (1, 1, 0, 0)
    assert executed == ["OSPF"]
