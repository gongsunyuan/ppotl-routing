"""M2-3 runner/评估管线产物契约测试（spec_runner.md #9）。

四组（全部 Abilene 目标 + episodes=2 + eval_interval=1 + cpu + config 注入
merge(load_config(), {...})）：A1 冒烟（产物全套+幂等）、A3 冒烟（预训练源加载）、
OSPF/A0 评估行、FAILED 路径（traceback、无 DONE）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from trl_sb3.common.config import load_config, merge
from trl_sb3.common.logging_utils import make_run_id
from trl_sb3.common.run_artifacts import is_done
from trl_sb3.env.routing_env import RoutingEnv
from trl_sb3.eval.ospf import run_ospf_eval
from trl_sb3.eval.zeroshot import run_zeroshot
from trl_sb3.train.pretrain import run_pretrain
from trl_sb3.train.runner import run_arm

TOPO = "Abilene.gml"
RATE = 500
STEPS_PER_EP = 50
EVAL_EPISODES = 5
COLUMNS = ["arm", "topo", "rate", "seed", "episode", "step", "rd", "rp", "th", "r_mean"]


def _config(root: Path) -> dict[str, Any]:
    return merge(
        load_config(),
        {
            "adapt": {"episodes": 2},
            "eval": {"eval_interval": 1},
            "paths": {"runs_dir": str(root / "runs"), "ckpts_dir": str(root / "ckpts")},
        },
    )


def _rows(run_path: Path) -> list[dict[str, str]]:
    with (run_path / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(run_path: Path, name: str) -> dict[str, Any]:
    return json.loads((run_path / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg(tmp_path_factory: Any) -> dict[str, Any]:
    return _config(tmp_path_factory.mktemp("runner_contract"))


@pytest.fixture(scope="module")
def pre_run(cfg: dict[str, Any]) -> Path:
    """共享预训练源（CERNET 1 回合）：A3 与 A0 都从同一 ckpt 微调/评估（§5 配对设计口径）。"""
    return run_pretrain(seed=0, episodes=1, config=cfg)


def test_a1_smoke_full_artifacts_and_idempotent_skip(cfg: dict[str, Any]) -> None:
    """Given A1(Abilene, episodes=2)注入 config；When run_arm；Then 产物全套：
    metrics 表头+2×50 行、eval.json 曲线 2+终评、manifest 臂因子、DONE、ckpt 存在；
    二次调用 is_done 幂等跳过（行数不增）。"""
    run_path = run_arm("A1", topo=TOPO, rate=RATE, seed=0, episodes=2, config=cfg)
    assert is_done(run_path)
    rows = _rows(run_path)
    assert len(rows) == 2 * STEPS_PER_EP
    assert list(rows[0].keys()) == COLUMNS
    assert {r["arm"] for r in rows} == {"A1"}
    assert {r["topo"] for r in rows} == {TOPO}
    eval_json = _json(run_path, "eval.json")
    assert len(eval_json["curve"]) == 2  # eval_interval=1 × episodes=2 → 2 个周期点
    assert set(eval_json["final"]) == {"r_mean_mean", "rd_mean", "rp_mean", "th_mean"}
    manifest = _json(run_path, "manifest.json")
    assert manifest["arm"] == "A1"
    assert manifest["factors"] == {"pretrain": False, "freeze": False, "pbrs": False}
    assert manifest["source_run_id"] is None
    assert manifest["run_id"] == run_path.name
    assert Path(cfg["paths"]["ckpts_dir"], f"{manifest['run_id']}.zip").exists()
    assert (run_path / "metrics.csv").exists() and not (run_path / "FAILED").exists()
    again = run_arm("A1", topo=TOPO, rate=RATE, seed=0, episodes=2, config=cfg)
    assert again == run_path
    assert len(_rows(run_path)) == 2 * STEPS_PER_EP


def test_a3_loads_pretrained_source_and_finishes(
    cfg: dict[str, Any], pre_run: Path
) -> None:
    """Given 预训练源 ckpt（CERNET 41 节点，整批 2050）；When A3(Abilene) 微调；
    Then 加载+冻结成功跑完（整批重算为 550），manifest.source_run_id 非空、臂因子正确。"""
    run_path = run_arm("A3", topo=TOPO, rate=RATE, seed=0, source_run_id=pre_run.name, episodes=2, config=cfg)
    assert is_done(run_path)
    manifest = _json(run_path, "manifest.json")
    assert manifest["source_run_id"] == pre_run.name
    assert manifest["factors"] == {"pretrain": True, "freeze": True, "pbrs": False}
    assert len(_rows(run_path)) == 2 * STEPS_PER_EP
    assert Path(cfg["paths"]["ckpts_dir"], f"{manifest['run_id']}.zip").exists()


def test_ospf_and_a0_zeroshot_eval_rows(cfg: dict[str, Any], pre_run: Path) -> None:
    """Given OSPF 恒 action 策略与预训练源 ckpt；When run_ospf_eval / run_zeroshot；
    Then arm 列分别为 OSPF/A0、各 5 评估回合出数、A0 manifest 带 zero_shot 标记与源 run_id。"""
    ospf_path = run_ospf_eval(TOPO, RATE, config=cfg)
    assert is_done(ospf_path)
    ospf_rows = _rows(ospf_path)
    assert len(ospf_rows) == EVAL_EPISODES * STEPS_PER_EP
    assert {r["arm"] for r in ospf_rows} == {"OSPF"}
    assert {r["episode"] for r in ospf_rows} == {str(i) for i in range(EVAL_EPISODES)}
    assert _json(ospf_path, "eval.json")["curve"] == []

    a0_path = run_zeroshot(pre_run.name, TOPO, RATE, config=cfg)
    assert is_done(a0_path)
    a0_manifest = _json(a0_path, "manifest.json")
    assert a0_manifest["arm"] == "A0"
    assert a0_manifest["zero_shot"] is True
    assert a0_manifest["source_run_id"] == pre_run.name
    a0_rows = _rows(a0_path)
    assert len(a0_rows) == EVAL_EPISODES * STEPS_PER_EP
    assert {r["arm"] for r in a0_rows} == {"A0"}


def test_failed_run_writes_traceback_without_done(
    cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given env.step 抛异常（A1+坏 env 注入，seed 区分 run_id）；When run_arm；
    Then 异常 re-raise、FAILED 含 traceback 与异常信息、无 DONE/ckpt。"""
    def _boom(self: RoutingEnv, actions: Any) -> tuple[Any, ...]:
        raise RuntimeError("boom-step")

    monkeypatch.setattr(RoutingEnv, "step", _boom)
    with pytest.raises(RuntimeError, match="boom-step"):
        run_arm("A1", topo=TOPO, rate=RATE, seed=99, episodes=1, config=cfg)
    expected_id = make_run_id(arm="A1", topo=TOPO, rate=RATE, seed=99, pbrs=False, freeze=False, pretrain=None)
    failed_dir = Path(cfg["paths"]["runs_dir"], expected_id)
    assert failed_dir.is_dir()
    failed_text = (failed_dir / "FAILED").read_text(encoding="utf-8")
    assert "RuntimeError" in failed_text and "boom-step" in failed_text
    assert not (failed_dir / "DONE").exists()


def test_interrupted_rerun_truncates_stale_metrics(cfg: dict[str, Any]) -> None:
    """Given 中断态 run 目录（预写 999 行垃圾 metrics.csv、无 DONE）；When run_arm
    同参数复跑；Then 旧垃圾行被截断（truncate-on-restart，issues.md M3）：新
    metrics.csv 全部行 arm=A1、行数=1×50，DONE 出现。"""
    run_id = make_run_id(arm="A1", topo=TOPO, rate=RATE, seed=77, pbrs=False, freeze=False, pretrain=None)
    run_path = Path(cfg["paths"]["runs_dir"], run_id)
    run_path.mkdir(parents=True, exist_ok=True)
    garbage = "\n".join(f"garbage,{i}" for i in range(999)) + "\n"
    (run_path / "metrics.csv").write_text(garbage, encoding="utf-8")  # 中断残留，无 DONE
    rerun = run_arm("A1", topo=TOPO, rate=RATE, seed=77, episodes=1, config=cfg)
    assert rerun == run_path
    rows = _rows(run_path)
    assert len(rows) == 1 * STEPS_PER_EP
    assert {r["arm"] for r in rows} == {"A1"}
    assert is_done(run_path)
