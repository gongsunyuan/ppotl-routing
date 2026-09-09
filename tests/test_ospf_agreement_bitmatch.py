"""OSPF-agreement 观察缝 on/off 位同 determinism 测试（计划 §5 叙事增项，EXPLORATORY）。

计划红线："纯观察不动 RNG，需 on/off 位同 determinism 测试"。同 run_id 两次
训练+评估（仅 eval.ospf_agreement_log 开/关不同，run 根目录隔离）：metrics.csv
逐字节相等、eval.json 除新增顶层 ospf_agreement 字段外逐字节相等、agreement
值 ∈ [0,1]——证明观察不消费 RNG、不动动作选择/seeding/回合流。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trl_sb3.common.config import load_config, merge
from trl_sb3.train.runner import run_arm

TOPO = "Abilene.gml"
RATE = 500
EVAL_EPISODES = 10  # D1 裁决：终评回合数全行型统一 10


def _config(root: Path, agreement_log: bool) -> dict[str, Any]:
    return merge(
        load_config(),
        {
            "adapt": {"episodes": 2},
            "eval": {"eval_interval": 1, "ospf_agreement_log": agreement_log},
            "paths": {"runs_dir": str(root / "runs"), "ckpts_dir": str(root / "ckpts")},
        },
    )


@pytest.fixture(scope="module")
def run_pair(tmp_path_factory: Any) -> tuple[Path, Path]:
    """同 run_id 两次 A1 冒烟（Abilene，2 回合，eval_interval=1）：关/开两根目录。"""
    root_off = tmp_path_factory.mktemp("agreement_off")
    root_on = tmp_path_factory.mktemp("agreement_on")
    off = run_arm(
        "A1", topo=TOPO, rate=RATE, seed=0, episodes=2, config=_config(root_off, False)
    )
    on = run_arm(
        "A1", topo=TOPO, rate=RATE, seed=0, episodes=2, config=_config(root_on, True)
    )
    return off, on


def test_flag_not_a_run_id_factor_same_directory_name(
    run_pair: tuple[Path, Path],
) -> None:
    """Given 仅 eval.ospf_agreement_log 不同的两次 run；When 比较 run 目录名；
    Then run_id 相同（观察开关不是 run_id 因素，run_id 确定性红线）。"""
    off, on = run_pair
    assert off.name == on.name


def test_metrics_csv_bit_identical_on_off(run_pair: tuple[Path, Path]) -> None:
    """Given 关/开两次同 run_id 训练+评估；When 逐字节读 metrics.csv；Then 完全
    相等——训练轨迹与评估回合流不受观察影响。"""
    off, on = run_pair
    assert (off / "metrics.csv").read_bytes() == (on / "metrics.csv").read_bytes()


def test_eval_json_identical_except_agreement_field(
    run_pair: tuple[Path, Path],
) -> None:
    """Given 同上；When 解析两侧 eval.json；Then OFF 侧仅 {curve, final}（final 恰
    四聚合键，纯附加证明）；ON 侧 == OFF 侧 + 顶层 ospf_agreement，无其他差异。"""
    off, on = run_pair
    off_json = json.loads((off / "eval.json").read_text(encoding="utf-8"))
    on_json = json.loads((on / "eval.json").read_text(encoding="utf-8"))
    assert set(off_json) == {"curve", "final"}
    assert set(off_json["final"]) == {"r_mean_mean", "rd_mean", "rp_mean", "th_mean"}
    assert set(on_json) == {"curve", "final", "ospf_agreement"}
    assert on_json["curve"] == off_json["curve"]
    assert on_json["final"] == off_json["final"]


def test_agreement_value_in_unit_interval(run_pair: tuple[Path, Path]) -> None:
    """Given ON 侧 eval.json 的 ospf_agreement；When 读取；Then per_episode 长度
    =终评回合数 10、逐值与 overall 均 ∈ [0,1]（贪心动作==ospf_action 的比例）。"""
    _, on = run_pair
    agreement = json.loads((on / "eval.json").read_text(encoding="utf-8"))[
        "ospf_agreement"
    ]
    assert len(agreement["per_episode"]) == EVAL_EPISODES
    assert all(0.0 <= value <= 1.0 for value in agreement["per_episode"])
    assert 0.0 <= agreement["overall"] <= 1.0
