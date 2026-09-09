"""tune 驱动器（计划 §2 工程缝 4）——调参试验的反腐蚀编排层。

计划 §7 点名单一最大风险："config/代码漂移 + 目录级 DONE 跳过 = 混配置表静默
腐蚀"。本驱动让每个调参产物可归属到精确 config+代码状态，并使阶段碰撞不可能：
① tag 派生编码阶段与集数（`<cfg>_a2000`/`<cfg>_b8000`，**防 Stage-B 被 Stage-A
的 DONE 静默跳过**——run_id 无超参因素，同目录不同阶段必撞）；② 碰撞守卫：已
存在 tag + overlay/grid/commit/协议哈希任一不同 → 拒绝，全同 → 试验级幂等跳过，
有目录无 trial_meta → 保守拒绝；③ 逐试验子进程执行（缝 6a：受控并发池
`_execute_pool` 以 `_execute_trial` 为调度原子，`--concurrency` 缺省 = §2.6 画像
实测饱和点，1 = 串行；并发只影响执行，聚合行序恒为 trial 定义序）；④ 聚合
summary.csv（确定性覆写）。

Trial YAML schema（`load_trials` 唯一入口；文件根为 list 或 `{trials: [...]}`，
亦可传内联 list[dict]）::

    trials:
      - name: lr2e-5_dual          # 必填；[A-Za-z0-9][A-Za-z0-9_.-]*（tag 即目录名）
        stage: a                   # 必填；a | b | 0（Stage-A/B/锚点）
        episodes: 2000             # 必填；预训练回合计数（预算，编码进 tag）
        seeds: [100]               # 必填；非空预训练种子列表
        grid:                      # 可选；sweep 网格主体（scenarios/arms/zeroshot/...）
          scenarios: [{topo: Claranet.gml, avgrate: 500}]
          seeds: [100]
          zeroshot: true
        config_overlay:            # 可选；缝 1 深合并 overlay（ppo/net/freeze/...）
          ppo: {learning_rate: 2.0e-5, lr_critic_ratio: 0.1}

tag 恒为 `f"{name}_{stage}{episodes}"` 派生：手写 `tag` 键拒绝，未知键拒绝（防
拼写静默漂移）；trial 级 seeds/episodes 压过 grid.pretrain；overlay.paths 强制
重定向 `<tune_root>/<tag>/{runs,ckpts}`（隔离红线 4，不可协商）。

trial_meta.json（执行前落盘）守卫字段：overlay_hash / grid_hash（含种子与集数，
同 tag 换种子 → 拒绝）/ git_commit / created_config_hash（=config_hash(合成后
merged config)，与路径无关——config_hash 只取 ppo/net/freeze/eval 协议节）。

summary.csv：status 在 dry-run 为 fresh|skip、实跑为 ok|failed|skipped；counts 由
plan_runs+DONE/FAILED 复推（skipped=planned−done−failed=未执行）；dev 指标取
trial 根下全部 DONE 的 A0 run（manifest arm=="A0"）的 eval.json final
r_mean_mean；dev_detail 形如 "Claranet.gml@500=0.612345"（确定性排序）。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trl_sb3.common.config import load_config, resolve_path
from trl_sb3.common.logging_utils import write_manifest
from trl_sb3.common.run_artifacts import config_hash, git_state, is_done
from trl_sb3.run.sweep import _merged_config, plan_runs, task_run_id

# allow: SIZE_OK — 本文件 ~55 行为任务契约交付物 docstring（YAML schema+守卫契约，须随文件走）；纯逻辑 ≈215 LOC，拆分会让 tune import 面目全非。
_ALLOWED_TRIAL_KEYS = frozenset(
    {"name", "stage", "episodes", "seeds", "grid", "config_overlay"}
)
_TRIAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_GUARD_FIELDS = (
    "tag",
    "overlay_hash",
    "grid_hash",
    "git_commit",
    "created_config_hash",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "tag",
    "name",
    "stage",
    "episodes",
    "seeds",
    "status",
    "exit_code",
    "planned",
    "done",
    "skipped",
    "failed",
    "n_a0",
    "a0_r_mean_mean",
    "dev_detail",
)

# —— §2.6 GPU 利用率画像定案（2026-08-30，证据 profiling/report.md）——
# 并发缺省 = cpu 路线实测饱和点：稳态（200ep 任务）聚合吞吐 N16=36.1 / N32=50.2 /
# N48=62.9 eps/s，N32→48 边际 +25%（>5% 继续扩）；N64 不取——20ep 电池边际仅
# +1.3%（<5% 停）且共享机需留核。cuda 路线仅宽度 ≤8 时按 run 占优（N8 每 run
# 2.83 vs cpu 2.35 eps/s），宽度 ≥16 反超输 cpu——tune 缺省走 cpu 宽池（runbook §6）。
# worker 线程配比 = 1：N48 下 OMP=2 聚合 32.0 vs OMP=1 的 38.0（20ep 电池），
# 缺省 32 线程在该宽度直接瘫痪（>25min 未完成）——64 核上多 worker × 多线程
# 互相争抢反降吞吐。红线：并发只影响执行，不影响任何选择准则；线程数不改数值
# （8 回合 CPU 对照 eval.json/metrics.csv 位同，见 report §threads-numeric）。
DEFAULT_CONCURRENCY = 48
DEFAULT_WORKER_THREADS = 1


class TrialSpecError(ValueError):
    """trial 定义不合法（未知键/手写 tag/坏 stage/name/episodes/seeds/重复 tag）。"""


class TrialCollisionError(RuntimeError):
    """碰撞守卫拒绝：同 tag 目录与当前 overlay/代码/协议状态不一致（防混配置表）。"""


@dataclass(frozen=True, slots=True)
class Trial:
    """一次调参试验的完整定义（tag 派生，见模块 docstring）。"""

    name: str
    stage: str | int
    episodes: int
    seeds: tuple[int, ...]
    grid: dict[str, Any]
    overlay: dict[str, Any]

    @property
    def tag(self) -> str:
        """派生 tag：f"{name}_{stage}{episodes}"——阶段+集数编码是防碰撞契约。"""
        return f"{self.name}_{self.stage}{self.episodes}"


@dataclass(frozen=True, slots=True)
class TrialResult:
    """单试验结果行（summary.csv 的内存形态）。"""

    tag: str
    name: str
    stage: str | int
    episodes: int
    seeds: tuple[int, ...]
    status: str  # dry-run: fresh|skip；实跑: ok|failed|skipped
    exit_code: int | None
    planned: int
    done: int
    skipped: int
    failed: int
    dev_rows: tuple[tuple[str, int, float], ...]  # (topo, rate, A0 final r_mean_mean)


def _canonical_hash(payload: Any) -> str:
    """sha256 全 hex over canonical JSON（与 run_artifacts.config_hash 同规范）。"""
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_trial(raw: Any, index: int) -> Trial:
    if not isinstance(raw, dict):
        raise TrialSpecError(f"trial[{index}] 必须为映射，got {type(raw).__name__}")
    if "tag" in raw:
        raise TrialSpecError('tag 必须派生（f"{name}_{stage}{episodes}"），不得手写')
    unknown = set(raw) - _ALLOWED_TRIAL_KEYS
    if unknown:
        raise TrialSpecError(
            f"trial[{index}] 未知键 {sorted(unknown)}；允许键 {sorted(_ALLOWED_TRIAL_KEYS)}"
        )
    name = raw.get("name")
    if not isinstance(name, str) or not _TRIAL_NAME_RE.fullmatch(name):
        raise TrialSpecError(
            f"trial[{index}].name 非法 {name!r}：须匹配 {_TRIAL_NAME_RE.pattern}"
        )
    stage = raw.get("stage")
    if isinstance(stage, bool) or stage not in ("a", "b", 0):
        raise TrialSpecError(f"trial[{index}].stage 非法 {stage!r}：只允许 a | b | 0")
    episodes = raw.get("episodes")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise TrialSpecError(f"trial[{index}].episodes 非法 {episodes!r}：须正整数")
    seeds = raw.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds)
    ):
        raise TrialSpecError(f"trial[{index}].seeds 非法 {seeds!r}：须非空整数列表")
    grid = raw.get("grid") or {}
    overlay = raw.get("config_overlay") or {}
    if not isinstance(grid, dict):
        raise TrialSpecError(f"trial[{index}].grid 须为映射")
    if not isinstance(overlay, dict):
        raise TrialSpecError(f"trial[{index}].config_overlay 须为映射")
    return Trial(name, stage, episodes, tuple(seeds), dict(grid), dict(overlay))


def load_trials(source: str | Path | Sequence[dict[str, Any]]) -> list[Trial]:
    """解析 trial 定义（YAML 文件路径或内联 dict 列表）→ 校验后的 Trial 列表。"""
    if isinstance(source, (str, Path)):
        doc = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        entries = doc.get("trials") if isinstance(doc, dict) else doc
        if not isinstance(entries, list):
            raise TrialSpecError(
                f"trials 文件根须为 list 或 {{'trials': [...]}}：{source}"
            )
    else:
        entries = list(source)
    trials = [_parse_trial(raw, index) for index, raw in enumerate(entries)]
    tags = [trial.tag for trial in trials]
    if len(set(tags)) != len(tags):
        dupes = sorted({tag for tag in tags if tags.count(tag) > 1})
        raise TrialSpecError(f"重复 tag {dupes}：同 tag 共享 trial_root，必然碰撞")
    return trials


def synthesize_grid(trial: Trial, trial_root: Path) -> dict[str, Any]:
    """trial → sweep 网格：trial 级 seeds/episodes 压过 grid.pretrain；overlay
    强制重定向 paths 到 trial_root/{runs,ckpts}（隔离红线，压过用户 overlay.paths）。"""
    pretrain = {
        **(trial.grid.get("pretrain") or {}),
        "seeds": list(trial.seeds),
        "episodes": trial.episodes,
    }
    overlay = {
        **trial.overlay,
        "paths": {
            **(trial.overlay.get("paths") or {}),
            "runs_dir": str(trial_root / "runs"),
            "ckpts_dir": str(trial_root / "ckpts"),
        },
    }
    return {**trial.grid, "pretrain": pretrain, "config_overlay": overlay}


def trial_meta(trial: Trial, base_cfg: dict[str, Any]) -> dict[str, Any]:
    """碰撞守卫用的试验指纹。created_config_hash 与 trial_root 取值无关
    （config_hash 只取协议节），守卫跨路径稳定。"""
    synth = synthesize_grid(trial, Path("<tune_root>") / trial.tag)
    definition = {
        "pretrain": {"seeds": list(trial.seeds), "episodes": trial.episodes},
        "grid": trial.grid,
    }
    return {
        "tag": trial.tag,
        "name": trial.name,
        "stage": trial.stage,
        "episodes": trial.episodes,
        "seeds": list(trial.seeds),
        "overlay_hash": _canonical_hash(trial.overlay),
        "grid_hash": _canonical_hash(definition),
        "git_commit": git_state()[0],
        "created_config_hash": config_hash(_merged_config(synth, base_cfg)),
    }


def check_collision(trial_root: Path, current: dict[str, Any]) -> str:
    """守卫：目录无 meta → fresh；meta 全同 → skip；任一守卫字段不一致或有目录
    无 meta / meta 不可解析 → TrialCollisionError 点名双方取值（保守拒绝）。"""
    meta_path = trial_root / "trial_meta.json"
    if not trial_root.exists():
        return "fresh"
    if not meta_path.exists():
        raise TrialCollisionError(
            f"碰撞守卫拒绝：{trial_root} 已存在但缺 trial_meta.json（来历不明）——人工确认后删除重跑"
        )
    try:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrialCollisionError(
            f"碰撞守卫拒绝：{meta_path} 不可解析（{exc}）——保守拒绝"
        ) from exc
    for field in _GUARD_FIELDS:
        if existing.get(field) != current.get(field):
            raise TrialCollisionError(
                f"碰撞守卫拒绝 tag={current.get('tag')}：trial_meta 字段 {field} 不一致 — "
                + f"已存在={existing.get(field)!r} 当前={current.get(field)!r}；"
                + f"不同 overlay/grid/代码/协议状态禁止复用同 tag 目录，确认后手动删除 {trial_root} 重跑"
            )
    return "skip"


def _worker_env() -> dict[str, str]:
    """并发 worker 子进程环境：强制 OMP/MKL 线程 = 实测配比（§2.6 决策 c，64 核机上
    缺省 32 线程 × N worker 互相争抢反降聚合吞吐）。线程数不改数值——CPU 对照实验
    eval.json/metrics.csv 位同（profiling/report.md §threads-numeric）。"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(DEFAULT_WORKER_THREADS)
    env["MKL_NUM_THREADS"] = str(DEFAULT_WORKER_THREADS)
    return env


def _execute_trial(command: Sequence[str]) -> int:
    """子进程执行单试验——受控并发池的调度原子（缝 6a 落点，docstring 预留缝）。
    非零退出只记 failed 不拖垮兄弟（子进程独立，池继续调度其余命令）。"""
    return subprocess.run(list(command), check=False, env=_worker_env()).returncode


def _execute_pool(commands: Sequence[Sequence[str]], concurrency: int) -> list[int]:
    """受控并发执行 fresh 试验：有界线程池逐命令 spawn 子进程，返回**与输入同序**的
    退出码（保序 = summary.csv 行序仍是 trial 定义序）。concurrency<=1 时退化为
    今日串行逐个执行；失败隔离：单命令崩溃只体现为非零退出码。"""
    if concurrency <= 1:
        return [_execute_trial(command) for command in commands]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(_execute_trial, commands))


def _aggregate(
    synth: dict[str, Any], base_cfg: dict[str, Any]
) -> tuple[int, int, int, int]:
    """复推 counts：planned=plan_runs 数；done/FAILED 按 run 目录标记；skipped=未执行。"""
    tasks = plan_runs(synth, base_cfg)
    runs_root = resolve_path(_merged_config(synth, base_cfg)["paths"]["runs_dir"])
    done = failed = 0
    for task in tasks:
        run_path = runs_root / task_run_id(task)
        if is_done(run_path):
            done += 1
        elif (run_path / "FAILED").exists():
            failed += 1
    planned = len(tasks)
    return planned, done, planned - done - failed, failed


def _collect_dev_rows(runs_root: Path) -> tuple[tuple[str, int, float], ...]:
    """trial 根下全部 DONE 的 A0 run → (topo, rate, final r_mean_mean)，确定性排序。"""
    rows: list[tuple[str, int, float]] = []
    if not runs_root.is_dir():
        return tuple(rows)
    for run_path in sorted(runs_root.iterdir()):
        if not is_done(run_path):
            continue
        manifest_path, eval_path = run_path / "manifest.json", run_path / "eval.json"
        if not (manifest_path.exists() and eval_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("arm") != "A0":
            continue
        final = json.loads(eval_path.read_text(encoding="utf-8"))["final"]
        rows.append(
            (str(manifest["topo"]), int(manifest["rate"]), float(final["r_mean_mean"]))
        )
    return tuple(sorted(rows))


def write_summary(path: Path, results: Sequence[TrialResult]) -> None:
    """确定性覆写 summary.csv（无时间戳；行序=trial 序；dev 行排序见 _collect_dev_rows）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, restval="")
        writer.writeheader()
        for result in results:
            means = [value for _, _, value in result.dev_rows]
            writer.writerow(
                {
                    "tag": result.tag,
                    "name": result.name,
                    "stage": result.stage,
                    "episodes": result.episodes,
                    "seeds": ";".join(str(s) for s in result.seeds),
                    "status": result.status,
                    "exit_code": "" if result.exit_code is None else result.exit_code,
                    "planned": result.planned,
                    "done": result.done,
                    "skipped": result.skipped,
                    "failed": result.failed,
                    "n_a0": len(result.dev_rows),
                    "a0_r_mean_mean": f"{sum(means) / len(means):.6f}" if means else "",
                    "dev_detail": ";".join(
                        f"{topo}@{rate}={value:.6f}"
                        for topo, rate, value in result.dev_rows
                    ),
                }
            )


def run_tune(
    trials: str | Path | Sequence[dict[str, Any]],
    *,
    device: str = "cpu",
    dry_run: bool = False,
    tune_root: str | Path = "runs_tune",
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[TrialResult]:
    """编排全部 trial：解析 → 预检守卫（不一致即拒，未烧任何算力）→ 受控并发执行
    （缝 6a：`_execute_trial` 为池的调度原子，只影响执行不影响选择；fresh 试验先
    全部落 trial_meta 再启动子进程——防中途崩溃留下无指纹目录）→ 按定义序聚合
    summary.csv（dry_run 不执行不落盘，只打印计划命令——freeze-gate 式验证）。"""
    if concurrency < 1:
        raise ValueError(f"concurrency 须 ≥1（got {concurrency}；1 = 串行）")
    base_cfg = load_config()
    root = resolve_path(tune_root)
    planned: list[tuple[Trial, dict[str, Any], dict[str, Any], str]] = []
    for trial in load_trials(trials):  # 预检守卫：任一 trial 碰撞即拒，未烧任何算力
        meta = trial_meta(trial, base_cfg)
        planned.append(
            (
                trial,
                synthesize_grid(trial, root / trial.tag),
                meta,
                check_collision(root / trial.tag, meta),
            )
        )
    print(
        f"[tune] trials={len(planned)} dry_run={dry_run} device={device} root={root} concurrency={concurrency}"
    )
    if dry_run:
        results: list[TrialResult] = []
        for index, (trial, synth, meta, verdict) in enumerate(planned, start=1):
            command = [
                sys.executable,
                "-m",
                "trl_sb3.run",
                "sweep",
                "--grid",
                "<grid.yaml>",
                "--device",
                device,
            ]
            print(
                f"[tune {index}/{len(planned)}] tag={trial.tag} {'skip' if verdict == 'skip' else 'fresh'} PLAN {' '.join(command)}"
            )
            planned_n, done_n, skipped_n, failed_n = _aggregate(synth, base_cfg)
            dev_rows = _collect_dev_rows(
                resolve_path(_merged_config(synth, base_cfg)["paths"]["runs_dir"])
            )
            results.append(
                TrialResult(
                    tag=trial.tag,
                    name=trial.name,
                    stage=trial.stage,
                    episodes=trial.episodes,
                    seeds=trial.seeds,
                    status=verdict,
                    exit_code=None,
                    planned=planned_n,
                    done=done_n,
                    skipped=skipped_n,
                    failed=failed_n,
                    dev_rows=dev_rows,
                )
            )
        return results
    # 实跑：先落全部 fresh 试验的指纹+临时网格，再并发执行，最后按定义序聚合（保序）
    with tempfile.TemporaryDirectory(prefix="tune-") as tmp:
        commands: list[list[str]] = []
        fresh_indices: list[int] = []
        for index, (trial, synth, meta, verdict) in enumerate(planned):
            if verdict == "skip":
                print(
                    f"[tune {index + 1}/{len(planned)}] tag={trial.tag} skipped PLAN (trial_meta 一致)"
                )
                continue
            trial_root = root / trial.tag
            write_manifest(trial_root / "trial_meta.json", meta)  # 先落指纹再启动子进程
            grid_yaml = Path(tmp) / f"grid_{trial.tag}.yaml"
            grid_yaml.write_text(
                yaml.safe_dump(synth, sort_keys=False), encoding="utf-8"
            )
            command = [
                sys.executable,
                "-m",
                "trl_sb3.run",
                "sweep",
                "--grid",
                str(grid_yaml),
                "--device",
                device,
            ]
            commands.append(command)
            fresh_indices.append(index)
            print(
                f"[tune {index + 1}/{len(planned)}] tag={trial.tag} fresh EXEC {' '.join(command)}"
            )
        exit_codes = _execute_pool(commands, concurrency)
    code_by_index = dict(zip(fresh_indices, exit_codes, strict=True))
    results = []
    for index, (trial, synth, meta, verdict) in enumerate(planned):
        if verdict == "skip":
            status, exit_code = "skipped", None
        else:
            exit_code = code_by_index[index]
            status = "ok" if exit_code == 0 else "failed"
        planned_n, done_n, skipped_n, failed_n = _aggregate(synth, base_cfg)
        dev_rows = _collect_dev_rows(
            resolve_path(_merged_config(synth, base_cfg)["paths"]["runs_dir"])
        )
        results.append(
            TrialResult(
                tag=trial.tag,
                name=trial.name,
                stage=trial.stage,
                episodes=trial.episodes,
                seeds=trial.seeds,
                status=status,
                exit_code=exit_code,
                planned=planned_n,
                done=done_n,
                skipped=skipped_n,
                failed=failed_n,
                dev_rows=dev_rows,
            )
        )
        print(
            f"[tune {index + 1}/{len(planned)}] tag={trial.tag} status={status} "
            + f"exit={exit_code if exit_code is not None else '-'} planned={planned_n} done={done_n} "
            + f"skipped={skipped_n} failed={failed_n} a0={len(dev_rows)}"
        )
    write_summary(root / "summary.csv", results)
    tally = {
        s: sum(1 for r in results if r.status == s) for s in ("ok", "failed", "skipped")
    }
    print(f"[tune] summary: {tally} → {root / 'summary.csv'}")
    return results
