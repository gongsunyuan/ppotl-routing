"""最优检查点 + loss 记录契约测试（DEVIATIONS #18）。

覆盖：峰值保存与终评取峰（final == 对 best_model.zip 的重评估）、血统 ckpt
即峰值模型、无评估点回落末态、losses.csv 列契约与批次数、并列峰值保持首见
（monkeypatch 剧本）、重跑 metrics 逐位一致（机制不扰动训练随机流）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import torch as th
from stable_baselines3 import PPO

from trl_sb3.common.config import load_config, merge
from trl_sb3.eval.evaluate import EVAL_AGG_KEYS
from trl_sb3.train.runner import LOSS_COLUMNS, run_arm


def _cfg(tmp_path: Path, **eval_over: Any) -> dict[str, Any]:
    return merge(
        load_config(),
        {
            "paths": {
                "runs_dir": str(tmp_path / "runs"),
                "ckpts_dir": str(tmp_path / "ckpts"),
            },
            "eval": {"eval_episodes": 2, **eval_over},
        },
    )


def _run(
    tmp_path: Path, episodes: int, *, eval_interval: int = 1, arm: str = "A1"
) -> Path:
    return run_arm(
        arm,
        topo="Abilene.gml",
        rate=500,
        seed=0,
        episodes=episodes,
        out_root=tmp_path / "runs",
        device="cpu",
        config=_cfg(tmp_path, eval_interval=eval_interval),
    )


def test_best_ckpt_final_is_reloaded_best(tmp_path: Path) -> None:
    """Given eval_interval=1 的 4 回合 A1；When run_arm；Then best = 曲线峰值回合、
    final == 对 best_model.zip 的重评估（加载峰值的终评口径）、final_last 留档末态。"""
    d = _run(tmp_path, episodes=4)
    ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
    assert len(ev["curve"]) == 4
    assert ev["best"] is not None
    assert ev["best"]["episode"] == float(
        max(ev["curve"], key=lambda p: p["r_mean_mean"])["episode"]
    )
    model = PPO.load(str(d / "best_model.zip"), device="cpu")
    from trl_sb3.eval.evaluate import greedy_eval, ppo_policy

    re_eval = greedy_eval(ppo_policy(model), "Abilene.gml", 500, config=_cfg(tmp_path))
    assert ev["final"]["r_mean_mean"] == re_eval["r_mean_mean"]
    assert "final_last" in ev  # 旧口径留档


def test_lineage_ckpt_equals_best_model(tmp_path: Path) -> None:
    """Given 同一 run；When 对比 ckpts_dir/<run_id>.zip 与 best_model.zip；Then
    policy 参数逐位相等——血统 ckpt 即峰值模型（迁移臂加载的就是峰值）。"""
    d = _run(tmp_path, episodes=4)
    run_id = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["run_id"]
    best = PPO.load(str(d / "best_model.zip"), device="cpu")
    lineage = PPO.load(str(tmp_path / "ckpts" / f"{run_id}.zip"), device="cpu")
    for (ka, va), (kb, vb) in zip(
        best.policy.state_dict().items(),
        lineage.policy.state_dict().items(),
        strict=True,
    ):
        assert ka == kb
        assert th.equal(va, vb)


def test_no_eval_point_falls_back_to_last(tmp_path: Path) -> None:
    """Given episodes=2 < eval_interval=100（无周期评估点）；When run_arm；Then
    best=None、final == final_last（回落末态）、不产 best_model.zip/losses 仍有行。"""
    d = _run(tmp_path, episodes=2, eval_interval=100)
    ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
    assert ev["best"] is None
    assert ev["final"] == ev["final_last"]
    assert not (d / "best_model.zip").exists()


def test_losses_csv_contract(tmp_path: Path) -> None:
    """Given 3 回合 run；When 读 losses.csv；Then 列恰为 LOSS_COLUMNS、行数 ==
    更新次数（n_steps=50 → 每回合一次更新）、关键损失列全部有限值。"""
    d = _run(tmp_path, episodes=3)
    with open(d / "losses.csv") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == LOSS_COLUMNS
        rows = list(reader)
    assert len(rows) == 3
    for row in rows:
        assert all(
            th.tensor(float(row[k])).isfinite().item()
            for k in ("policy_gradient_loss", "value_loss", "entropy_loss")
        )
    assert [float(r["update"]) for r in rows] == [1.0, 2.0, 3.0]


def test_best_tiebreak_first_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given monkeypatch 剧本评估 [0.5, 0.9, 0.9]（2、3 回合并列峰值）；When
    run_arm；Then best.episode == 2（严格大于才更新，并列保持首见）。"""
    script = [
        {"r_mean_mean": 0.5, "rd_mean": 0.5, "rp_mean": 0.5, "th_mean": 0.5},
        {"r_mean_mean": 0.9, "rd_mean": 0.9, "rp_mean": 0.9, "th_mean": 0.9},
        {"r_mean_mean": 0.9, "rd_mean": 0.9, "rp_mean": 0.9, "th_mean": 0.9},
        {
            "r_mean_mean": 0.4,
            "rd_mean": 0.4,
            "rp_mean": 0.4,
            "th_mean": 0.4,
        },  # final_last（末态）
        {
            "r_mean_mean": 0.8,
            "rd_mean": 0.8,
            "rp_mean": 0.8,
            "th_mean": 0.8,
        },  # final（峰值重评估）
    ]
    calls = iter(script)

    def fake_greedy_eval(
        policy: Any, topo: str, rate: int, *, config: dict[str, Any]
    ) -> dict[str, float]:
        return {k: float(v) for k, v in next(calls).items() if k in EVAL_AGG_KEYS}

    monkeypatch.setattr("trl_sb3.train.runner.greedy_eval", fake_greedy_eval)
    d = _run(tmp_path, episodes=3)
    ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
    assert [p["r_mean_mean"] for p in ev["curve"]] == [0.5, 0.9, 0.9]
    assert ev["best"] is not None and ev["best"]["episode"] == 2.0
    assert ev["final_last"]["r_mean_mean"] == 0.4
    assert ev["final"]["r_mean_mean"] == 0.8  # 峰值检查点经剧本第 5 次评估


def test_rerun_metrics_bit_identical(tmp_path: Path) -> None:
    """Given 同 spec 两次独立 run（不同 tmp 根）；When 对比 metrics.csv；Then
    逐字节相等——best-ckpt/losses 机制不消耗训练随机流（重跑可复现旧轨迹）。"""
    d1 = _run(tmp_path / "a", episodes=3)
    d2 = _run(tmp_path / "b", episodes=3)
    assert (d1 / "metrics.csv").read_bytes() == (d2 / "metrics.csv").read_bytes()
