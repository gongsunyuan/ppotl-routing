"""产物契约原语（M2-3）：DONE/FAILED 标记、manifest 组装、评估行落盘。

契约（根 AGENTS.md 约束 3 / 计划 §6）：每 run 目录写 metrics.csv / eval.json /
manifest.json / DONE|FAILED；DONE 最后写（聚合端只扫 DONE 目录）；FAILED 带
traceback 全文；时间戳只进 manifest.created_at，不进 run_id 与文件名。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx
import numpy as np
import stable_baselines3
import torch

from trl_sb3.common.config import PKG_ROOT
from trl_sb3.common.logging_utils import MetricsCSVWriter

# metrics.csv 长表列契约（计划 §6）：rd/rp/th 为 per-node 均值标量。
METRICS_COLUMNS: tuple[str, ...] = (
    "arm", "topo", "rate", "seed", "episode", "step", "rd", "rp", "th", "r_mean",
)

# config_hash 只取协议承载节（计划 §2 工程缝 4 "paper_defaults+eval 协议哈希"）：
# 预算（pretrain/adapt episodes）与路径不是协议，不入哈希。
CONFIG_HASH_SECTIONS: tuple[str, ...] = ("ppo", "net", "freeze", "eval")


def config_hash(config: Mapping[str, Any]) -> str:
    """run 的协议哈希（sha256 全 hex）——**规范化契约，跨机器可复现**：

    1. 取 merged config 中在场的协议节：``{k: config[k] for k in ("ppo","net","freeze","eval") if k in config}``
       （缺节剔除不补 null——容忍部分 config / key-set 演化）；
    2. canonical JSON：``json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``；
    3. utf-8 编码后 sha256 hexdigest（64 hex）。

    键集或键值任一变化 → 哈希必变；pretrain.episodes / paths 变化 → 哈希不变。
    """
    payload = {key: config[key] for key in CONFIG_HASH_SECTIONS if key in config}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_git(*args: str) -> str | None:
    """git 子命令输出（成功时 stdout strip）；git 缺失/超时/非零 → None，绝不抛。"""
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False  # 固定参数调用本机 git
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_state(repo_root: str | Path | None = None) -> tuple[str | None, bool | None]:
    """(HEAD commit 40-hex, dirty) 经 subprocess 读取；仓库外/git 不可用优雅降级
    (None, None)——run 绝不因 git 崩（计划 §2 工程缝 4）。"""
    root = PKG_ROOT if repo_root is None else Path(repo_root)
    commit = _run_git("-C", str(root), "rev-parse", "HEAD")
    if commit is None:
        return None, None
    status = _run_git("-C", str(root), "status", "--porcelain")
    return commit, None if status is None else bool(status)


def is_done(run_dir: str | Path) -> bool:
    """run 是否已完成（DONE 文件存在）——断点续跑/聚合端的唯一跳过依据。"""
    return (Path(run_dir) / "DONE").exists()


def mark_done(run_dir: str | Path) -> None:
    """最后写 DONE（此前产物若中断则缺 DONE，聚合端自然跳过）。"""
    (Path(run_dir) / "DONE").write_text("", encoding="utf-8")


def mark_failed(run_dir: str | Path, exc: BaseException) -> None:
    """FAILED 写 traceback 全文（排查依据；与 DONE 互斥）。"""
    (Path(run_dir) / "FAILED").write_text("".join(traceback.format_exception(exc)), encoding="utf-8")


def build_manifest(
    run_id: str,
    arm: str,
    topo: str,
    rate: int,
    seed: int,
    factors: dict[str, bool],
    source_run_id: str | None,
    episodes: int,
    device: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装 manifest：臂因子、config 快照全文、requirements.lock 全文（experiments 根）、
    关键依赖版本、溯源三件套 config_hash/git_commit/git_dirty（工程缝 4；仓库外
    git_*=None）、git_hash（旧占位键，恒 None）、created_at ISO 时间戳。extra 顶层合入。"""
    commit, dirty = git_state(PKG_ROOT)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "arm": arm,
        "topo": topo,
        "rate": int(rate),
        "seed": int(seed),
        "factors": dict(factors),
        "source_run_id": source_run_id,
        "episodes": int(episodes),
        "device": str(device),
        "config": config,
        "config_hash": config_hash(config),
        "git_commit": commit,
        "git_dirty": dirty,
        "requirements_lock": (PKG_ROOT / "requirements.lock").read_text(encoding="utf-8"),
        "git_hash": None,
        "git_hash_note": "无 git 仓库",
        "versions": {
            "torch": torch.__version__,
            "sb3": stable_baselines3.__version__,
            "numpy": np.__version__,
            "networkx": networkx.__version__,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_eval_rows(
    csv_writer: MetricsCSVWriter,
    arm: str,
    topo: str,
    rate: int,
    seed: int,
    ep_idx: int,
    episodes_result: Mapping[str, Any],
) -> None:
    """把 greedy_eval 单回合 steps 写成 metrics 行（每步一行，episode=评估回合序号）。"""
    for step_row in episodes_result["steps"]:
        csv_writer.write_row(
            {
                "arm": arm,
                "topo": topo,
                "rate": rate,
                "seed": seed,
                "episode": ep_idx,
                "step": step_row["step"],
                "rd": step_row["rd"],
                "rp": step_row["rp"],
                "th": step_row["th"],
                "r_mean": step_row["r_mean"],
            }
        )
