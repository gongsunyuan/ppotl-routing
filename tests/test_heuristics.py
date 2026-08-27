"""M2.5 启发式基线测试（计划 §5/Q5 裁决）。

四组：split 形状/归一化约束（含 LB 首步=ECMP 退化、负载反比生效）、
RR 逐流轮转、rollout 冒烟（三策略各一回合全有限）、run_heuristic_eval
产物契约（arm 列 / 10×50 行 / eval.json 聚合有限 / manifest / DONE / 幂等）。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trl_sb3.common.config import load_config, merge
from trl_sb3.common.envs import build_routing_env
from trl_sb3.common.run_artifacts import is_done
from trl_sb3.env.routing_env import RoutingEnv
from trl_sb3.eval.heuristics import (
    HEURISTIC_SPLIT_FNS,
    ecmp_split,
    heuristic_eval,
    lb_split,
    rollout_split_episode,
    rr_split,
    run_heuristic_eval,
)

TOPO = "Abilene.gml"
RATE = 500
COLUMNS = ["arm", "topo", "rate", "seed", "episode", "step", "rd", "rp", "th", "r_mean"]


def _fresh_env() -> RoutingEnv:
    """reset(seed=首评估种子) 后的干净 eval env（pbrs=False 基口径，DEVIATIONS #12）。"""
    env = build_routing_env(TOPO, RATE, pbrs=False, seed=10000, config=load_config())
    env.reset(seed=10000)
    return env


def _assert_split_contract(split: np.ndarray, env: RoutingEnv) -> None:
    assert split.shape == (env._n, env._n_candidates)
    assert np.all(split >= 0.0)
    assert np.allclose(split.sum(axis=1), 1.0)


def test_split_shapes_and_row_normalization() -> None:
    """Given fresh env（reset 后 _in_rate 空）；When 三策略 split；
    Then (N,K) 非负、行和=1；LB 首步无负载退化为 ECMP。"""
    env = _fresh_env()
    for split_fn in (ecmp_split, lb_split, rr_split):
        _assert_split_contract(split_fn(env, 0), env)
    assert np.allclose(lb_split(env, 0), ecmp_split(env, 0))


def test_lb_uses_last_step_load() -> None:
    """Given 已注入一步流量的 env（_in_rate 非空）；When lb_split；
    Then 负载反比生效（≠等分）且行仍归一化。"""
    env = _fresh_env()
    env._add_flows(ecmp_split(env, 0))
    split = lb_split(env, 1)
    _assert_split_contract(split, env)
    assert not np.allclose(split, ecmp_split(env, 1))


def test_rr_rotates_onehot_position_per_step() -> None:
    """Given fresh env；When rr_split 连续步；Then one-hot 位置 = step mod n_paths 逐流轮转。"""
    env = _fresh_env()
    for step in range(4):
        split = rr_split(env, step)
        assert np.allclose(split.sum(axis=1), 1.0)
        assert np.allclose(split.max(axis=1), 1.0)
        for i, paths in enumerate(env._flow_paths):
            assert int(split[i].argmax()) == step % len(paths)


@pytest.mark.parametrize("arm", ["ECMP", "LB", "RR"])
def test_rollout_smoke_all_finite(arm: str) -> None:
    """Given 三策略之一；When rollout_split_episode 一回合 50 步；
    Then 步序 0..49 齐全、rd/rp/th/r_mean 与聚合键全有限。"""
    env = _fresh_env()
    result = rollout_split_episode(env, HEURISTIC_SPLIT_FNS[arm], 50)
    assert [s["step"] for s in result["steps"]] == list(range(50))
    for key in ("rd", "rp", "th", "r_mean"):
        assert all(math.isfinite(s[key]) for s in result["steps"]), key
    for key in ("r_mean_sum", "rd_mean", "rp_mean", "th_mean"):
        assert math.isfinite(result[key]), key


def test_heuristic_eval_unknown_arm_raises() -> None:
    """Given 未知臂名；When heuristic_eval；Then ValueError（臂查表先于 env 构造）。"""
    with pytest.raises(ValueError, match="unknown heuristic arm"):
        heuristic_eval("XX", TOPO, RATE, episodes=1)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory: Any) -> dict[str, Any]:
    return merge(
        load_config(),
        {
            "paths": {
                "runs_dir": str(tmp_path_factory.mktemp("heur_runs") / "runs"),
                "ckpts_dir": str(tmp_path_factory.mktemp("heur_ckpts") / "ckpts"),
            }
        },
    )


def _rows(run_path: Path) -> list[dict[str, str]]:
    with (run_path / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("arm", ["ECMP", "LB", "RR"])
def test_run_heuristic_eval_artifact_contract(arm: str, cfg: dict[str, Any]) -> None:
    """Given 注入 tmp runs_dir 的 config；When run_heuristic_eval；
    Then arm 列正确、10×50=500 行、eval.json final 聚合有限、manifest 因子/回合数、
    DONE、二次调用幂等跳过（行数不增）。"""
    run_path = run_heuristic_eval(arm, TOPO, RATE, config=cfg)
    assert is_done(run_path)
    rows = _rows(run_path)
    assert len(rows) == 10 * 50
    assert list(rows[0].keys()) == COLUMNS
    assert {r["arm"] for r in rows} == {arm}
    assert {r["topo"] for r in rows} == {TOPO}
    assert {r["episode"] for r in rows} == {str(i) for i in range(10)}
    eval_json = json.loads((run_path / "eval.json").read_text(encoding="utf-8"))
    assert eval_json["curve"] == []
    assert set(eval_json["final"]) == {"r_mean_mean", "rd_mean", "rp_mean", "th_mean"}
    assert all(math.isfinite(v) for v in eval_json["final"].values())
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["arm"] == arm
    assert manifest["episodes"] == 10
    assert manifest["factors"] == {"pretrain": False, "freeze": False, "pbrs": False}
    assert manifest["source_run_id"] is None
    assert manifest["run_id"] == run_path.name
    again = run_heuristic_eval(arm, TOPO, RATE, config=cfg)
    assert again == run_path
    assert len(_rows(run_path)) == 10 * 50
