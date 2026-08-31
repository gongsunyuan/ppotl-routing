"""超参传播缝测试（experiment-execution-plan §2 缝 2）：per-run config 的
net.layers / freeze 覆盖必须真正到达 policy 构建与冻结，而非被全局
load_config() 静默吞掉。

覆盖：默认 config 架构表征（改前改后都过——回落路径未被破坏）、
_build_model 注入覆盖后的网络形状（修前 RED：静默忽略 bug 的直接证据）、
policy_kwargs 经 SB3 ckpt save/load 往返存活、net_layers 输入维校验
（注入路径与 config 回落路径同口径）、apply_freeze 显式 freeze_cfg
注入与无 cfg 回落表征、runner 冻结臂消费注入 freeze_cfg。

SB3 2.7.0 往返机制（venv 源码核对）：save() pickle model.__dict__（含
policy_kwargs，base_class.py save→_excluded_save_params 不排除它）→
load() __dict__.update(data) 后 _setup_model() 以
``**self.policy_kwargs`` 重建 policy（on_policy_algorithm.py）——普通
tuple/float 可 pickle，往返天然成立；policy 子类只需在 super().__init__
前 pop 自有键（基类签名定长，未知键 TypeError）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from test_sb3_smoke import SEED, _layer_shapes, _make_vec

from trl_sb3.common.config import load_config, merge
from trl_sb3.policy.freeze import apply_freeze
from trl_sb3.policy.policy import ActorCriticPolicy
from trl_sb3.train.runner import _build_model, _RunSpec

TOPO = "Abilene.gml"
RATE = 500
DEFAULT_HIDDEN_SHAPES = [(128, 287), (64, 128), (3, 64)]  # default.yaml net.layers [287,128,64]


def _vec() -> VecMonitor:
    return VecMonitor(_make_vec(TOPO, SEED))


def _spec(config: dict[str, Any], *, freeze: bool = False, out_root: Path) -> _RunSpec:
    """_build_model 直调用 spec（不训练，只验构建期传播）。"""
    return _RunSpec(
        arm="A3b" if freeze else "A1",
        topo=TOPO,
        rate=RATE,
        seed=0,
        pbrs=False,
        freeze=freeze,
        source_run_id=None,
        episodes=1,
        out_root=out_root,
        ckpts_dir=out_root / "ckpts",
        device="cpu",
        config=config,
    )


def _policy(**policy_kwargs: Any) -> ActorCriticPolicy:
    """独立 policy（Abilene 空间）；manual_seed 固定构造期全局 rng 消耗。"""
    vec = _make_vec(TOPO, SEED)
    try:
        th.manual_seed(SEED)
        return ActorCriticPolicy(
            vec.observation_space, vec.action_space, lr_schedule=lambda _: 6.0e-6, **policy_kwargs
        )
    finally:
        vec.close()


def test_default_config_architecture_characterization() -> None:
    """Given 不传 policy_kwargs 的 policy；When 构建；Then 隐层 = default.yaml
    net.layers [287,128,64] 的 [(128,287),(64,128),(3,64)]（无覆盖回落路径表征，
    改前改后都必须过——默认行为逐位不变的锚点）。"""
    policy = _policy()
    assert _layer_shapes(policy.mlp_extractor.actor) == DEFAULT_HIDDEN_SHAPES
    assert _layer_shapes(policy.mlp_extractor.critic) == [(128, 287), (64, 128), (1, 64)]


def test_build_model_propagates_net_layers_override(tmp_path: Path) -> None:
    """Given net.layers 覆盖为 [287,32,16] 的注入 config；When _build_model；
    Then policy 隐层形状反映覆盖 [(32,287),(16,32),(3,16)]——修前这里 RED：
    policy 读全局 load_config() 出 [(128,287),...]，覆盖被静默忽略（本缝要修的 bug）。"""
    cfg = merge(load_config(), {"net": {"layers": [287, 32, 16]}})
    vec = _vec()
    try:
        model = _build_model(vec, _spec(cfg, out_root=tmp_path), tmp_path / "ckpts")
        assert isinstance(model.policy, ActorCriticPolicy)
        assert _layer_shapes(model.policy.mlp_extractor.actor) == [(32, 287), (16, 32), (3, 16)]
        assert _layer_shapes(model.policy.mlp_extractor.critic) == [(32, 287), (16, 32), (1, 16)]
    finally:
        vec.close()


def test_policy_kwargs_net_layers_shape_and_default_optimizer() -> None:
    """Given net_layers=(287,64,64)+lr_critic_ratio=0.1 的 policy_kwargs；When 直接
    构造 policy；Then 形状按注入构网、ratio 存为属性并生效为双参数组 Adam（缝 3
    起消费：两组 lr=[6e-6, 6e-7]，critic/actor=0.1 复刻 legacy actor/critic=10×，
    DEVIATIONS #3）；defaults lr 仍 6.0e-6（actor 基值）。"""
    policy = _policy(net_layers=(287, 64, 64), lr_critic_ratio=0.1)
    assert _layer_shapes(policy.mlp_extractor.actor) == [(64, 287), (64, 64), (3, 64)]
    assert policy.lr_critic_ratio == 0.1
    assert isinstance(policy.optimizer, th.optim.Adam)
    assert [g["lr"] for g in policy.optimizer.param_groups] == pytest.approx([6.0e-6, 6.0e-7])
    assert policy.optimizer.defaults["lr"] == 6.0e-6


def test_policy_kwargs_survive_sb3_ckpt_roundtrip(tmp_path: Path) -> None:
    """Given net_layers/lr_critic_ratio 覆盖构建的 PPO；When save→PPO.load；
    Then 加载侧 policy 形状同、lr_critic_ratio 仍为 0.1（policy_kwargs 被 SB3
    pickle 进 zip 并在 load 时重新 splat 进 policy 构造）。"""
    vec = _vec()
    try:
        model = PPO(
            policy=ActorCriticPolicy,
            env=vec,
            n_steps=50,
            batch_size=50 * 11,
            learning_rate=6.0e-6,
            policy_kwargs={"net_layers": (287, 64, 64), "lr_critic_ratio": 0.1},
            seed=0,
            device="cpu",
        )
        path = str(tmp_path / "override_model")
        model.save(path)
        reloaded = PPO.load(path, device="cpu")
        assert isinstance(reloaded.policy, ActorCriticPolicy)
        assert _layer_shapes(reloaded.policy.mlp_extractor.actor) == [(64, 287), (64, 64), (3, 64)]
        assert reloaded.policy.lr_critic_ratio == 0.1
        assert tuple(reloaded.policy_kwargs["net_layers"]) == (287, 64, 64)
    finally:
        vec.close()


def test_build_model_default_config_bit_identical_to_direct_construction(tmp_path: Path) -> None:
    """Given default config（无覆盖）；When _build_model 与等参直接 PPO 构造（无
    policy_kwargs 的旧口径）；Then 两者 policy state_dict 逐位相等——缝的硬约束：
    默认路径架构/优化器/RNG 消耗逐位不变（policy_kwargs 恒注入不改变行为）。"""
    cfg = load_config()
    vec_a, vec_b = _vec(), _vec()
    try:
        via_runner = _build_model(vec_a, _spec(cfg, out_root=tmp_path), tmp_path / "ckpts")
        ppo_cfg = cfg["ppo"]
        direct = PPO(
            policy=ActorCriticPolicy,
            env=vec_b,
            n_steps=ppo_cfg["n_steps"],
            n_epochs=ppo_cfg["n_epochs"],
            batch_size=ppo_cfg["n_steps"] * 11,
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            learning_rate=ppo_cfg["learning_rate"],
            normalize_advantage=ppo_cfg["normalize_advantage"],
            verbose=0,
            seed=0,
            device="cpu",
        )
        for (ka, va), (kb, vb) in zip(
            via_runner.policy.state_dict().items(), direct.policy.state_dict().items(), strict=True
        ):
            assert ka == kb
            assert th.equal(va, vb)
    finally:
        vec_a.close()
        vec_b.close()


def test_net_layers_input_dim_validation() -> None:
    """Given 输入维与观测不符的 net_layers=(999,64)；When 构造 policy；Then
    ValueError（注入路径与 config 回落路径同一校验口径）。"""
    with pytest.raises(ValueError, match="999"):
        _policy(net_layers=(999, 64))


def test_apply_freeze_uses_injected_freeze_cfg() -> None:
    """Given 显式 freeze_cfg（只冻 actor.1、不重置头）；When apply_freeze(freeze_cfg=...)；
    Then 仅 actor[1] requires_grad=False、头与 critic 可训练且头参数逐位不变
    （不读全局 default.yaml 的 [actor.0-3]+reinit_head=true）。"""
    policy = _policy()
    head_before = {n: p.detach().clone() for n, p in policy.named_parameters() if n.startswith("mlp_extractor.actor.4.")}
    apply_freeze(
        policy,
        freeze_cfg={"frozen_layers": ["actor.1"], "reinit_head": False, "critic_trainable": True},
    )
    actor = policy.mlp_extractor.actor
    assert all(not p.requires_grad for p in actor[1].parameters())
    for idx in (0, 2, 4):
        assert all(p.requires_grad for p in actor[idx].parameters())
    assert all(p.requires_grad for p in policy.mlp_extractor.critic.parameters())
    for name, param in policy.named_parameters():
        if name.startswith("mlp_extractor.actor.4."):
            assert th.equal(head_before[name], param.detach())


def test_apply_freeze_no_cfg_falls_back_to_default_config() -> None:
    """Given 不传 freeze_cfg；When apply_freeze()；Then 读 default.yaml freeze 节
    （actor.0-3 全冻结，回落表征：既有调用方/测试口径不变）。"""
    policy = _policy()
    apply_freeze(policy)
    actor = policy.mlp_extractor.actor
    for idx in (0, 1, 2, 3):
        assert all(not p.requires_grad for p in actor[idx].parameters())
    assert all(p.requires_grad for p in actor[-1].parameters())
    assert all(p.requires_grad for p in policy.mlp_extractor.critic.parameters())


def test_build_model_applies_injected_freeze_cfg(tmp_path: Path) -> None:
    """Given freeze 节覆盖为只冻 actor.1 的注入 config + freeze 臂；When
    _build_model；Then 冻结模式来自注入 cfg 而非全局 default.yaml（修前 RED：
    apply_freeze 读全局，注入被静默忽略，actor.0 也会被冻结）。"""
    cfg = merge(load_config(), {"freeze": {"frozen_layers": ["actor.1"], "reinit_head": False}})
    vec = _vec()
    try:
        model = _build_model(vec, _spec(cfg, freeze=True, out_root=tmp_path), tmp_path / "ckpts")
        assert isinstance(model.policy, ActorCriticPolicy)
        actor = model.policy.mlp_extractor.actor
        assert all(not p.requires_grad for p in actor[1].parameters())
        assert all(p.requires_grad for p in actor[0].parameters())
    finally:
        vec.close()
