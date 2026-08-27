"""A0 零样本行（M2-3，F6）：加载源 ckpt **不训练**直接贪心评估。

= legacy main_trans 的真实语义（其"跨拓扑"实为纯零样本评估，从未微调——
DEVIATIONS #5）；manifest 记 zero_shot=true + 源 run_id（lineage）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from trl_sb3.common.config import load_config, resolve_path
from trl_sb3.eval.evaluate import ppo_policy, run_eval_only


def run_zeroshot(
    source_run_id: str,
    topo: str,
    rate: int,
    out_root: str | Path | None = None,
    seed: int = 0,
    device: str = "cpu",
    config: dict[str, Any] | None = None,
) -> Path:
    """零样本评估源 ckpt 于目标场景（arm="A0"），返回 run 目录。"""
    cfg = load_config() if config is None else config
    ckpt = resolve_path(cfg["paths"]["ckpts_dir"]) / f"{source_run_id}.zip"
    model = PPO.load(ckpt, device=device)
    return run_eval_only(
        "A0",
        topo,
        rate,
        policy_fn=ppo_policy(model),
        seed=seed,
        out_root=out_root,
        config=cfg,
        extra_manifest={"zero_shot": True, "pretrain": source_run_id, "device": device},
    )
