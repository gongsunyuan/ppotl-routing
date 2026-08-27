"""环境工厂：从 config 构建 RoutingEnv（M2-3）。

论文超参全部经 config env/ppo 节映射（根 AGENTS.md 约束 1），其余参数走
RoutingEnv 默认（legacy 常数）。评估与训练共用此入口，保证口径一致。
"""

from __future__ import annotations

from typing import Any

from trl_sb3.common.config import load_config, resolve_path
from trl_sb3.env.routing_env import RoutingEnv


def build_routing_env(
    topo: str, avgrate: int, *, pbrs: bool, seed: int, config: dict[str, Any] | None = None
) -> RoutingEnv:
    """按 config 构建 RoutingEnv：env 节映射环境参数 + ppo 节 gamma；
    gml = resolve_path(topologies/<topo>)；未列参数用 RoutingEnv 默认（legacy 常数）。"""
    cfg = load_config() if config is None else config
    env_cfg = cfg["env"]
    return RoutingEnv(
        resolve_path(f"topologies/{topo}"),
        avgrate=avgrate,
        alpha_plr=env_cfg["alpha_plr"],
        beta_delay=env_cfg["beta_delay"],
        pbrs=pbrs,
        seed=seed,
        max_nodes=env_cfg["max_nodes"],
        n_candidates=env_cfg["K"],
        capacity=env_cfg["capacity"],
        mu_choices=tuple(env_cfg["mu"]),
        gamma=cfg["ppo"]["gamma"],
        mix_local=env_cfg["reward_mix"]["local"],
        mix_global=env_cfg["reward_mix"]["global"],
        episode_steps=env_cfg["episode_steps"],
    )
