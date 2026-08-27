"""预训练 runner（M2-3）：完整方法口径的源域训练（A5/A6 之源）。

裁决（spec_runner #5）：预训练域 **pbrs=True**（完整方法口径）；A2/A3 微调时环境
pbrs=0——PBRS 是消融因子，只随臂因子表进目标域训练 env。DEVIATIONS 已记。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trl_sb3.common.config import load_config
from trl_sb3.train.runner import _RunSpec, _train_and_write


def run_pretrain(
    seed: int,
    episodes: int | None = None,
    out_root: str | Path | None = None,
    device: str = "cpu",
    config: dict[str, Any] | None = None,
) -> Path:
    """源域预训练（topo/rate 读 config pretrain 节），返回 run 目录。"""
    cfg = load_config() if config is None else config
    pretrain_cfg = cfg["pretrain"]
    return _train_and_write(
        _RunSpec(
            arm="pretrain",
            topo=pretrain_cfg["topology"],
            rate=int(pretrain_cfg["avgrate"]),
            seed=seed,
            pbrs=True,
            freeze=False,
            source_run_id=None,
            episodes=int(pretrain_cfg["episodes"]) if episodes is None else int(episodes),
            out_root=Path(out_root) if out_root is not None else None,
            ckpts_dir=None,
            device=device,
            config=cfg,
        )
    )
