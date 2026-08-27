"""sweep 编排器（M2-4，决策 D5 lineage + 计划 §6）：网格 → 计划 → 顺序执行。

lineage 守卫：迁移臂（A2/A3/A5/A6）与 A0 启动前校验源 ckpt zip 与源 run DONE，
缺失则 SKIP-LINEAGE（--strict 抛错终止）；目标 run 已 DONE 则跳过（幂等续跑）；
单 run 异常捕获记 failed 继续，不拖垮 sweep；--dry-run 只打印计划清单（状态
NEW/SKIP-DONE/SKIP-LINEAGE）；--filter 按 arm/topo 子串过滤。顺序执行、无
multiprocessing（Windows spawn 守卫；M3/M4 长跑由用户自管进程）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trl_sb3.common.config import load_config, merge, resolve_path
from trl_sb3.common.logging_utils import make_run_id
from trl_sb3.common.run_artifacts import is_done
from trl_sb3.eval.heuristics import HEURISTIC_SPLIT_FNS, run_heuristic_eval
from trl_sb3.eval.ospf import run_ospf_eval
from trl_sb3.eval.zeroshot import run_zeroshot
from trl_sb3.train.pretrain import run_pretrain
from trl_sb3.train.runner import ARM_FACTORS, run_arm


@dataclass(frozen=True, slots=True)
class RunTask:
    """一次计划 run 的全部决定因素（config=grid 合后快照，随 run 进 manifest）。"""

    arm: str
    topo: str
    rate: int
    seed: int
    source_run_id: str | None
    episodes: int
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SweepSummary:
    """sweep 结束计数（planned=过滤后计划数；done=本次新完成）。"""

    planned: int
    done: int
    skipped_done: int
    skipped_lineage: int
    failed: int


def _merged_config(grid: dict[str, Any], base_cfg: dict[str, Any]) -> dict[str, Any]:
    """grid 预算字段并入 base config：pretrain.episodes / adapt.episodes。"""
    overrides: dict[str, Any] = {}
    pretrain_episodes = (grid.get("pretrain") or {}).get("episodes")
    if pretrain_episodes is not None:
        overrides["pretrain"] = {"episodes": int(pretrain_episodes)}
    if grid.get("adapt_episodes") is not None:
        overrides["adapt"] = {"episodes": int(grid["adapt_episodes"])}
    return merge(base_cfg, overrides)


def _pretrain_run_id(cfg: dict[str, Any], seed: int) -> str:
    """预训练 run_id 预计算（不查盘）：与 run_pretrain 实际产物目录名逐位一致——
    因素口径 arm=pretrain / pbrs=true / freeze=false / pretrain=none（pretrain.py）。"""
    return make_run_id(
        arm="pretrain",
        topo=cfg["pretrain"]["topology"],
        rate=int(cfg["pretrain"]["avgrate"]),
        seed=seed,
        pbrs=True,
        freeze=False,
        pretrain=None,
    )


def task_run_id(task: RunTask) -> str:
    """RunTask 的确定性 run_id 预计算——必须与底层 runner 落盘目录名逐位一致
    （训练臂因素查 ARM_FACTORS；A0/OSPF 走 run_eval_only 口径 pbrs/freeze=false）。"""
    if task.arm == "pretrain":
        return _pretrain_run_id(task.config, task.seed)
    if task.arm in HEURISTIC_SPLIT_FNS or task.arm in ("A0", "OSPF"):
        # eval-only 口径 pbrs/freeze=false；启发式恒 seed=0、source=None（与
        # run_heuristic_eval 的因素逐位一致），A0/OSPF 带各自 seed/source。
        return make_run_id(
            arm=task.arm,
            topo=task.topo,
            rate=task.rate,
            seed=task.seed,
            pbrs=False,
            freeze=False,
            pretrain=task.source_run_id,
        )
    factors = ARM_FACTORS[task.arm]
    return make_run_id(
        arm=task.arm,
        topo=task.topo,
        rate=task.rate,
        seed=task.seed,
        pbrs=factors["pbrs"],
        freeze=factors["freeze"],
        pretrain=task.source_run_id if factors["pretrain"] else None,
    )


def plan_runs(grid: dict[str, Any], base_cfg: dict[str, Any]) -> list[RunTask]:
    """网格 → RunTask 清单（纯函数，不查盘）。顺序：先全部预训练（seed s → 源），
    后全部臂行（arm×场景×种子）、A0 零样本行、OSPF 基线行、启发式基线行
    （heuristic 键）。grid 键说明见 config/grid_main.yaml 注释；A1b 总预算读
    a1b_episodes（缺省=预训练+适应）。"""
    cfg = _merged_config(grid, base_cfg)
    seeds = [int(s) for s in grid.get("seeds", [])]
    scenarios = [(str(sc["topo"]), int(sc["avgrate"])) for sc in grid.get("scenarios", [])]
    pretrain_node = grid.get("pretrain") or {}
    pretrain_seeds = [int(s) for s in pretrain_node.get("seeds", [])]
    pretrain_episodes = int(cfg["pretrain"]["episodes"])
    adapt_episodes = int(cfg["adapt"]["episodes"])
    a1b_episodes = (
        int(grid["a1b_episodes"]) if grid.get("a1b_episodes") is not None else pretrain_episodes + adapt_episodes
    )
    eval_episodes = int(cfg["eval"]["eval_episodes"])
    tasks: list[RunTask] = [
        RunTask(
            arm="pretrain",
            topo=str(cfg["pretrain"]["topology"]),
            rate=int(cfg["pretrain"]["avgrate"]),
            seed=seed,
            source_run_id=None,
            episodes=pretrain_episodes,
            config=cfg,
        )
        for seed in pretrain_seeds
    ]
    for arm in grid.get("arms", []):
        if arm not in ARM_FACTORS:
            raise ValueError(f"网格含未知消融臂 {arm!r}；可用：{sorted(ARM_FACTORS)}")
        for topo, rate in scenarios:
            for seed in seeds:
                source = _pretrain_run_id(cfg, seed) if ARM_FACTORS[arm]["pretrain"] else None
                episodes = a1b_episodes if arm == "A1b" else adapt_episodes
                tasks.append(
                    RunTask(arm=arm, topo=topo, rate=rate, seed=seed, source_run_id=source, episodes=episodes, config=cfg)
                )
    if grid.get("zeroshot", False):
        for topo, rate in scenarios:
            for seed in seeds:
                tasks.append(
                    RunTask(
                        arm="A0", topo=topo, rate=rate, seed=seed,
                        source_run_id=_pretrain_run_id(cfg, seed), episodes=eval_episodes, config=cfg,
                    )
                )
    if grid.get("ospf", False):
        # OSPF 的 run_id 无 seed 因素（run_ospf_eval 恒 seed=0）→ 每场景只计划 1 行。
        for topo, rate in scenarios:
            tasks.append(
                RunTask(arm="OSPF", topo=topo, rate=rate, seed=0, source_run_id=None, episodes=eval_episodes, config=cfg)
            )
    heuristics = [str(name) for name in grid.get("heuristic", [])]
    unknown_heuristics = [name for name in heuristics if name not in HEURISTIC_SPLIT_FNS]
    if unknown_heuristics:
        raise ValueError(f"网格含未知启发式 {unknown_heuristics}；可用：{sorted(HEURISTIC_SPLIT_FNS)}")
    if heuristics:
        # 启发式同 OSPF 口径：确定性分流、run_id 无 seed 因素（恒 seed=0）→ 每场景每策略 1 行；
        # 评估回合数独立读 eval.heuristic_episodes（§5 十评估种子协议）。
        heuristic_episodes = int(cfg["eval"]["heuristic_episodes"])
        for name in heuristics:
            for topo, rate in scenarios:
                tasks.append(
                    RunTask(arm=name, topo=topo, rate=rate, seed=0, source_run_id=None, episodes=heuristic_episodes, config=cfg)
                )
    return tasks


def _lineage_ok(source_run_id: str, runs_root: Path, ckpts_root: Path) -> bool:
    """源 lineage 完整：源 ckpt zip 存在且源 run 目录 DONE（D5：缺源拒绝启动迁移臂）。"""
    return (ckpts_root / f"{source_run_id}.zip").exists() and is_done(runs_root / source_run_id)


def _matches(task: RunTask, filters: str | Sequence[str] | None) -> bool:
    """--filter 子串过滤：任一子串命中 arm 或 topo 名即保留；无过滤全过。"""
    if not filters:
        return True
    needles = [filters] if isinstance(filters, str) else list(filters)
    return any(needle in task.arm or needle in task.topo for needle in needles)


def _execute(task: RunTask, device: str) -> Path:
    """分派单任务到底层 runner（顺序执行；产物契约与 FAILED 落盘由 runner 负责）。"""
    if task.arm == "pretrain":
        return run_pretrain(task.seed, episodes=task.episodes, device=device, config=task.config)
    if task.arm == "OSPF":
        return run_ospf_eval(task.topo, task.rate, config=task.config)
    if task.arm in HEURISTIC_SPLIT_FNS:
        return run_heuristic_eval(task.arm, task.topo, task.rate, config=task.config)
    if task.arm == "A0":
        return run_zeroshot(
            task.source_run_id or "", task.topo, task.rate, seed=task.seed, device=device, config=task.config
        )
    return run_arm(
        task.arm,
        topo=task.topo,
        rate=task.rate,
        seed=task.seed,
        source_run_id=task.source_run_id,
        episodes=task.episodes,
        device=device,
        config=task.config,
    )


def run_sweep(
    grid_path: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = True,
    filters: str | Sequence[str] | None = None,
    device: str = "cpu",
    strict: bool = False,
    base_config: dict[str, Any] | None = None,
) -> SweepSummary:
    """执行（或仅规划）一次 sweep，返回并打印 SweepSummary。

    守卫顺序（逐任务）：DONE 跳过 → lineage 校验 → 执行。dry_run 时计划内的预训练
    视为将来可用源（只规划不执行）；单 run 异常捕获记 failed 继续（--strict 仅
    作用于 lineage 拒绝）；base_config 注入测试用 config（缺省 default.yaml）。
    """
    grid = load_config(resolve_path(grid_path))
    base_cfg = load_config() if base_config is None else base_config
    cfg = _merged_config(grid, base_cfg)
    tasks = [task for task in plan_runs(grid, base_cfg) if _matches(task, filters)]
    runs_root = resolve_path(cfg["paths"]["runs_dir"])
    ckpts_root = resolve_path(cfg["paths"]["ckpts_dir"])
    planned_pretrain_ids = {task_run_id(task) for task in tasks if task.arm == "pretrain"}
    print(f"[sweep] grid={grid_path} tasks={len(tasks)} dry_run={dry_run} device={device} strict={strict}")
    if not dry_run:
        print("[sweep] Windows 长跑提醒（管理员 PowerShell）：powercfg /change standby-timeout-ac 0")
    done = skipped_done = skipped_lineage = failed = 0
    for index, task in enumerate(tasks, start=1):
        run_id = task_run_id(task)
        if resume and is_done(runs_root / run_id):
            status = "SKIP-DONE"
        elif (
            task.source_run_id is not None
            and not _lineage_ok(task.source_run_id, runs_root, ckpts_root)
            and not (dry_run and task.source_run_id in planned_pretrain_ids)
        ):
            status = "SKIP-LINEAGE"
        else:
            status = "NEW"
        print(
            f"[sweep {index}/{len(tasks)}] {status} arm={task.arm} topo={task.topo} "
            f"rate={task.rate} seed={task.seed} source={task.source_run_id or '-'} "
            f"episodes={task.episodes} run_id={run_id}"
        )
        if status == "SKIP-LINEAGE" and strict:
            raise RuntimeError(f"lineage 守卫拒绝（--strict）：{task.arm} 的源 {task.source_run_id} 缺 ckpt 或未 DONE")
        if dry_run or status != "NEW":
            if status == "SKIP-DONE":
                skipped_done += 1
            elif status == "SKIP-LINEAGE":
                skipped_lineage += 1
            continue
        try:
            _execute(task, device)
            done += 1
        except Exception as exc:  # noqa: BLE001 — sweep 边界：单 run 崩溃须隔离不拖垮其余 run；FAILED 已由 runner 落盘
            failed += 1
            print(f"[sweep] FAILED arm={task.arm} seed={task.seed} run_id={run_id}: {exc!r}")
    summary = SweepSummary(
        planned=len(tasks), done=done, skipped_done=skipped_done, skipped_lineage=skipped_lineage, failed=failed
    )
    print(
        f"[sweep] summary: planned={summary.planned} done={summary.done} "
        f"skipped_done={summary.skipped_done} skipped_lineage={summary.skipped_lineage} failed={summary.failed}"
    )
    return summary
