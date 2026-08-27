"""启发式基线（M2.5，计划 §5/Q5 裁决）：ECMP 等分 / LB 负载反比 / RR 轮转单路径。

不改 env（M0 冻结）：评估不走 RoutingEnv.step()（其动作 one-hot 化与比例分流
不相容），而是走 M0-2 规格固化的分流缝 ``env._add_flows(split)``——split 为
(N, K) 任意非负比例矩阵，step 的 one-hot 是其特例（routing_env.py:201-207）。
驱动循环 rollout_split_episode 与 step()（routing_env.py:179-199）逐行同构，
仅 split 来自策略而非 one-hot（语义出入记 notepads/issues.md）。
产物契约与 OSPF 行（ospf.py → run_eval_only）同格式；评估回合数独立于
eval.eval_episodes，读 config eval.heuristic_episodes（§5 十评估种子协议）。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

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
from trl_sb3.env.routing_env import RoutingEnv
from trl_sb3.eval.evaluate import EVAL_AGG_KEYS

# 分流策略签名：读 env 内部态（_flow_paths/_in_rate，M0-2 seam）+ 回合内步序 → (N,K) 比例矩阵。
SplitFn = Callable[[RoutingEnv, int], np.ndarray]


def ecmp_split(env: RoutingEnv, step_idx: int) -> np.ndarray:
    """ECMP 等分（§5）：每 flow 的全部候选路径均分 1/n_paths（step_idx 不参与）。"""
    split = np.zeros((env._n, env._n_candidates))
    for i, paths in enumerate(env._flow_paths):
        split[i, : len(paths)] = 1.0 / len(paths)
    return split


def lb_split(env: RoutingEnv, step_idx: int) -> np.ndarray:
    """LB 负载反比（§5）：路径负载度量 = 上一步各路径节点总入率
    sum(_in_rate[v].values()) 之和；权重 ∝ 1/(1+load_k)，行归一化。
    fresh env 首步 _in_rate 为空 → 等权 → 退化为 ECMP（step_idx 不参与）。"""
    split = np.zeros((env._n, env._n_candidates))
    has_load = bool(env._in_rate)  # reset 不清 _in_rate：回合 2+ 首步沿用上回合尾负载（记 issues）
    for i, paths in enumerate(env._flow_paths):
        loads = np.zeros(len(paths))
        if has_load:
            for k, path in enumerate(paths):
                loads[k] = sum(sum(env._in_rate[v].values()) for v in path)
        weights = 1.0 / (1.0 + loads)
        split[i, : len(paths)] = weights / weights.sum()
    return split


def rr_split(env: RoutingEnv, step_idx: int) -> np.ndarray:
    """RR 轮转单路径（§5）：保持 one-hot，每步轮换候选位置 step_idx mod n_paths。

    无内部计数器（步序经签名注入）→ 回合内轮转恒从位置 0 起，确定性可复现。"""
    split = np.zeros((env._n, env._n_candidates))
    for i, paths in enumerate(env._flow_paths):
        split[i, step_idx % len(paths)] = 1.0
    return split


HEURISTIC_SPLIT_FNS: dict[str, SplitFn] = {
    "ECMP": ecmp_split,
    "LB": lb_split,
    "RR": rr_split,
}


def rollout_split_episode(
    env: RoutingEnv, split_fn: SplitFn, episode_steps: int
) -> dict[str, Any]:
    """单回合比例分流驱动（调用方先 env.reset(seed)，同 greedy_eval 回合循环）。

    与 RoutingEnv.step（routing_env.py:179-199）逐行同构：_add_flows ← :184、
    _get_reward ← :185、th 先算后存 ← :186、_change_flows ← :187、_build_state ← :188
    （split 策略读 env 内部态，obs 不消费，仅保持调用序列完整）。省略项：
    one-hot 构造（:181-183，被比例 split 取代）与 _t += 1（:190，_t 仅被 step()
    自身的 truncated 判定读取，语义惰性——记 issues.md）。
    返回与 greedy_eval 单回合同构（seed 由调用方合入）。"""
    steps: list[dict[str, float]] = []
    for step in range(episode_steps):
        split = split_fn(env, step)
        env._add_flows(split)  # ← step():184
        reward = env._get_reward(split)  # ← step():185
        th = env._rates / env._th_max_rate  # ← step():186 N§4 先算后存
        env._change_flows(reward.total)  # ← step():187
        env._build_state()  # ← step():188
        steps.append(
            {
                "step": step,
                "rd": float(reward.rd.mean()),
                "rp": float(reward.rp.mean()),
                "th": float(th.mean()),
                "r_mean": float(reward.r_mean),
            }
        )
    return {
        "steps": steps,
        "r_mean_sum": float(sum(s["r_mean"] for s in steps)),
        "rd_mean": float(np.mean([s["rd"] for s in steps])),
        "rp_mean": float(np.mean([s["rp"] for s in steps])),
        "th_mean": float(np.mean([s["th"] for s in steps])),
    }


def heuristic_eval(
    arm: str,
    topo: str,
    rate: int,
    *,
    episodes: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """固定种子流比例分流评估（greedy_eval 的 split 版）：eval env pbrs=False、
    构造种子=首评估种子、每回合 reset(seed=eval_seed_base+i)；回合数独立——
    episodes 缺省读 config eval.heuristic_episodes。返回结构同 greedy_eval。"""
    cfg = load_config() if config is None else config
    try:
        split_fn = HEURISTIC_SPLIT_FNS[arm]
    except KeyError:
        raise ValueError(
            f"unknown heuristic arm: {arm!r} (expected one of {sorted(HEURISTIC_SPLIT_FNS)})"
        ) from None
    if episodes is None:
        episodes = int(cfg["eval"]["heuristic_episodes"])
    eval_seeds = [int(cfg["eval"]["eval_seed_base"]) + i for i in range(episodes)]
    steps_per_episode = int(cfg["env"]["episode_steps"])
    env = build_routing_env(topo, int(rate), pbrs=False, seed=eval_seeds[0], config=cfg)
    episodes_out: list[dict[str, Any]] = []
    for ep_seed in eval_seeds:
        env.reset(seed=ep_seed)
        episode = rollout_split_episode(env, split_fn, steps_per_episode)
        episodes_out.append({"seed": int(ep_seed), **episode})
    all_r_mean = [s["r_mean"] for episode in episodes_out for s in episode["steps"]]
    return {
        "episodes": episodes_out,
        "r_mean_mean": float(np.mean(all_r_mean)),
        "rd_mean": float(np.mean([e["rd_mean"] for e in episodes_out])),
        "rp_mean": float(np.mean([e["rp_mean"] for e in episodes_out])),
        "th_mean": float(np.mean([e["th_mean"] for e in episodes_out])),
    }


def run_heuristic_eval(
    name: str,
    topo: str,
    rate: int,
    out_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """启发式 eval-only run（arm=name∈{ECMP,LB,RR}，seed=0），与 run_ospf_eval
    同格式产物：metrics.csv 每评估步行 / eval.json curve=[]+final 聚合 /
    manifest（episodes=回合数，§5 协议）/ DONE 最后写 / FAILED 带 traceback /
    is_done 幂等跳过。"""
    cfg = load_config() if config is None else config
    run_id = make_run_id(
        arm=name, topo=topo, rate=int(rate), seed=0, pbrs=False, freeze=False, pretrain=None
    )
    root = Path(out_root) if out_root is not None else resolve_path(cfg["paths"]["runs_dir"])
    directory = run_dir(root, run_id)
    if is_done(directory):
        return directory
    try:
        result = heuristic_eval(name, topo, int(rate), config=cfg)
        with MetricsCSVWriter(directory / "metrics.csv", METRICS_COLUMNS) as writer:
            for ep_idx, episode in enumerate(result["episodes"]):
                write_eval_rows(writer, name, topo, int(rate), 0, ep_idx, episode)
        write_manifest(
            directory / "eval.json",
            {"curve": [], "final": {key: result[key] for key in EVAL_AGG_KEYS}},
        )
        manifest = build_manifest(
            run_id,
            name,
            topo,
            int(rate),
            0,
            factors={"pretrain": False, "freeze": False, "pbrs": False},
            source_run_id=None,
            episodes=len(result["episodes"]),
            device="cpu",
            config=cfg,
        )
        write_manifest(directory / "manifest.json", manifest)
        mark_done(directory)
    except Exception as exc:
        mark_failed(directory, exc)
        raise
    return directory
