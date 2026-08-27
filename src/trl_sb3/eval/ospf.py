"""OSPF 基线评估（M2-3，F10）：全节点恒取 config eval.ospf_action。

勘误口径（DEVIATIONS #8）：legacy 注释 "hop count" 有误，action 0 实为按 Σ1/μ
权重的 k-最短首路径——OSPF 行 = 恒 action 0 的贪心评估，与其他臂同协议同种子流。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from trl_sb3.common.config import load_config
from trl_sb3.eval.evaluate import run_eval_only


def ospf_policy(obs_batch: np.ndarray) -> np.ndarray:
    """全节点恒 action=config eval.ospf_action（F10；不读观测，形状对齐即可）。"""
    action = int(load_config()["eval"]["ospf_action"])
    return np.full(len(obs_batch), action, dtype=np.int64)


def run_ospf_eval(
    topo: str, rate: int, out_root: str | Path | None = None, config: dict[str, Any] | None = None
) -> Path:
    """OSPF eval-only run（arm="OSPF", seed=0），返回 run 目录。"""
    cfg = load_config() if config is None else config
    return run_eval_only("OSPF", topo, rate, policy_fn=ospf_policy, seed=0, out_root=out_root, config=cfg)
