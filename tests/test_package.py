"""包结构冒烟测试 + default.yaml（paper_defaults）契约锁定。"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import trl_sb3
from trl_sb3.common.config import PKG_ROOT, load_config, merge, resolve_path


def test_package_has_version() -> None:
    assert isinstance(trl_sb3.__version__, str)
    assert trl_sb3.__version__ == "0.1.0"


def test_six_subpackages_importable() -> None:
    for name in ("env", "policy", "train", "eval", "common", "run"):
        assert importlib.import_module(f"trl_sb3.{name}") is not None


def test_default_config_paper_defaults_contract() -> None:
    """Given 默认配置；When 加载；Then paper_defaults 关键超参与计划 D2/D3/§3 一致。"""
    cfg = load_config()
    assert cfg["env"]["K"] == 3
    assert cfg["env"]["max_nodes"] == 41
    assert cfg["env"]["capacity"] == 10000
    assert cfg["env"]["mu"] == [1000, 2000, 3000]
    assert cfg["env"]["alpha_plr"] == 0.7
    assert cfg["env"]["beta_delay"] == 0.3
    assert cfg["env"]["reward_mix"] == {"local": 0.6, "global": 0.4}
    assert cfg["env"]["episode_steps"] == 50
    assert cfg["ppo"]["n_steps"] == 50
    assert cfg["ppo"]["n_epochs"] == 10
    assert cfg["ppo"]["batch_size"] is None  # = 50·N 由 runner 按拓扑计算
    assert cfg["ppo"]["gamma"] == 0.99
    assert cfg["ppo"]["gae_lambda"] == 1.0
    assert cfg["ppo"]["clip_range"] == 0.2
    assert cfg["ppo"]["ent_coef"] == 0.05
    assert cfg["ppo"]["learning_rate"] == 6e-6
    assert cfg["ppo"]["normalize_advantage"] is True
    assert cfg["net"]["layers"] == [287, 128, 64]
    assert cfg["net"]["activation"] == "PReLU"
    assert cfg["net"]["separate_pi_vf"] is True
    assert cfg["eval"]["deterministic"] is True
    assert cfg["eval"]["eval_seed_base"] == 10000
    assert cfg["eval"]["eval_episodes"] == 5
    assert cfg["eval"]["ospf_action"] == 0
    assert cfg["pretrain"]["topology"] == "CERNET.gml"
    assert cfg["pretrain"]["avgrate"] == 500
    assert cfg["pretrain"]["episodes"] == 8000
    assert len(cfg["scenarios"]) == 4
    assert cfg["seeds"] == list(range(10))


def test_merge_deep_overrides() -> None:
    """Given 嵌套 base 与 override；When merge；Then 深覆盖、原 dict 不被改动。"""
    base = {"ppo": {"gamma": 0.99, "n_epochs": 10}, "env": {"K": 3}}
    override = {"ppo": {"gamma": 0.95}}
    merged = merge(base, override)
    assert merged == {"ppo": {"gamma": 0.95, "n_epochs": 10}, "env": {"K": 3}}
    assert base["ppo"]["gamma"] == 0.99  # 输入不可变


def test_resolve_path_against_package_root() -> None:
    """Given 相对/绝对路径；When resolve_path；Then 分别锚定包根 / 原样返回。"""
    relative = resolve_path("topologies/CERNET.gml")
    assert relative == PKG_ROOT / "topologies" / "CERNET.gml"  # 锚定包根，与检出目录名无关
    absolute_input = "C:/some/absolute/path.yaml" if os.name == "nt" else "/some/absolute/path.yaml"
    absolute = resolve_path(absolute_input)
    assert absolute == Path(absolute_input)
