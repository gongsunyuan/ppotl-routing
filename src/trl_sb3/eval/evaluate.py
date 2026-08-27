"""贪心评估管线（M2-3，决策 D8）：固定评估种子流、pbrs=False 基任务口径、
eval-only run 产物契约（OSPF / A0 零样本行 / 任何 policy_fn）。

口径钉死（spec_runner 关键协议）：
- 评估 env **pbrs=False**：基任务奖励跨臂可比（PBRS 是消融因子，只进训练 env）；
- 评估 env 与训练 env 严格分离；评估种子流 seed=10000+i（config eval 节）全臂
  全目标共享冻结；构造种子=首评估种子（mu/dst/rate_init 固定，跨臂全同）。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from trl_sb3.common.config import load_config, resolve_path
from trl_sb3.common.envs import build_routing_env
from trl_sb3.common.logging_utils import MetricsCSVWriter, make_run_id, run_dir, write_manifest
from trl_sb3.common.run_artifacts import (
    METRICS_COLUMNS,
    build_manifest,
    is_done,
    mark_done,
    mark_failed,
    write_eval_rows,
)

PolicyFn = Callable[[np.ndarray], np.ndarray]
# 聚合键：eval.json final / 曲线点共用的四元组。
EVAL_AGG_KEYS: tuple[str, ...] = ("r_mean_mean", "rd_mean", "rp_mean", "th_mean")


def ppo_policy(model: PPO) -> PolicyFn:
    """把 SB3 模型包成 PolicyFn；deterministic 读 config eval.deterministic（D8 贪心）。"""
    deterministic = bool(load_config()["eval"]["deterministic"])

    def _predict(obs: np.ndarray) -> np.ndarray:
        actions, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(actions)

    return _predict


def greedy_eval(
    policy_fn: PolicyFn,
    topo: str,
    avgrate: int,
    *,
    eval_seeds: list[int] | None = None,
    n_episodes: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """固定种子流贪心评估：每回合 reset(seed=eval_seeds[i]) 后贪心跑满 episode_steps。

    返回 {"episodes":[{seed, steps:[{step,rd,rp,th,r_mean}], r_mean_sum, rd_mean,
    rp_mean, th_mean}], *EVAL_AGG_KEYS}。rd/rp/th 为 per-node 均值标量。"""
    cfg = load_config() if config is None else config
    if n_episodes is None:
        n_episodes = int(cfg["eval"]["eval_episodes"])
    if eval_seeds is None:
        base = int(cfg["eval"]["eval_seed_base"])
        eval_seeds = [base + i for i in range(n_episodes)]
    steps_per_episode = int(cfg["env"]["episode_steps"])
    env = build_routing_env(topo, avgrate, pbrs=False, seed=eval_seeds[0], config=cfg)
    episodes: list[dict[str, Any]] = []
    for ep_seed in eval_seeds:
        obs, _ = env.reset(seed=ep_seed)
        steps: list[dict[str, float]] = []
        for step in range(steps_per_episode):
            obs, _, _, _, info = env.step(policy_fn(obs))
            steps.append(
                {
                    "step": step,
                    "rd": float(info["rd"].mean()),
                    "rp": float(info["rp"].mean()),
                    "th": float(info["th"].mean()),
                    "r_mean": float(info["r_mean"]),
                }
            )
        episodes.append(
            {
                "seed": int(ep_seed),
                "steps": steps,
                "r_mean_sum": float(sum(s["r_mean"] for s in steps)),
                "rd_mean": float(np.mean([s["rd"] for s in steps])),
                "rp_mean": float(np.mean([s["rp"] for s in steps])),
                "th_mean": float(np.mean([s["th"] for s in steps])),
            }
        )
    all_r_mean = [s["r_mean"] for episode in episodes for s in episode["steps"]]
    return {
        "episodes": episodes,
        "r_mean_mean": float(np.mean(all_r_mean)),
        "rd_mean": float(np.mean([e["rd_mean"] for e in episodes])),
        "rp_mean": float(np.mean([e["rp_mean"] for e in episodes])),
        "th_mean": float(np.mean([e["th_mean"] for e in episodes])),
    }


def run_eval_only(
    arm: str,
    topo: str,
    rate: int,
    *,
    policy_fn: PolicyFn,
    seed: int = 0,
    out_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    """eval-only run 产物契约：run_id 因素 arm/topo/rate/seed/pbrs=false/freeze=false/
    pretrain=extra 里给；metrics.csv=评估回合每步行；eval.json={"curve":[], "final":聚合}；
    manifest（extra 合入）；DONE 最后 / FAILED 带 traceback；is_done 幂等跳过。"""
    cfg = load_config() if config is None else config
    extra = dict(extra_manifest) if extra_manifest else {}
    run_id = make_run_id(
        arm=arm,
        topo=topo,
        rate=int(rate),
        seed=seed,
        pbrs=False,
        freeze=False,
        pretrain=extra.get("pretrain"),
    )
    root = Path(out_root) if out_root is not None else resolve_path(cfg["paths"]["runs_dir"])
    directory = run_dir(root, run_id)
    if is_done(directory):
        return directory
    try:
        result = greedy_eval(policy_fn, topo, int(rate), config=cfg)
        with MetricsCSVWriter(directory / "metrics.csv", METRICS_COLUMNS) as writer:
            for ep_idx, episode in enumerate(result["episodes"]):
                write_eval_rows(writer, arm, topo, int(rate), seed, ep_idx, episode)
        write_manifest(
            directory / "eval.json",
            {"curve": [], "final": {key: result[key] for key in EVAL_AGG_KEYS}},
        )
        manifest = build_manifest(
            run_id,
            arm,
            topo,
            int(rate),
            seed,
            factors={"pretrain": extra.get("pretrain") is not None, "freeze": False, "pbrs": False},
            source_run_id=extra.get("pretrain"),
            episodes=int(cfg["eval"]["eval_episodes"]),
            device="cpu",
            config=cfg,
            extra=extra,
        )
        write_manifest(directory / "manifest.json", manifest)
        mark_done(directory)
    except Exception as exc:
        mark_failed(directory, exc)
        raise
    return directory
