"""缝 3（双 lr）测试：lr_critic_ratio 消费面——policy._build 双参数组 Adam、
freeze.rebuild_optimizer 镜像、双 lr ckpt → PPO.load → 真实 adapt 往返。

legacy 方向（已钉死，DEVIATIONS #3 / PLAN.md F13）：actor Adam lr 6e-6（wd 1e-4
已弃不复刻）+ critic Adam lr 6e-7 → actor lr = 10 × critic lr，即
lr_critic_ratio = critic_lr / actor_lr = 0.1 复刻「actor/critic=10×」。

SB3 2.7.0 坑（venv 源码核对）：PPO.train() 每次先 _update_learning_rate
（ppo.py:191 → common/utils.py update_learning_rate）把**所有** param_groups 的
lr 重置为同一 schedule 基值——纯 th.optim.Adam 双组的 critic lr 会被静默抬到
actor 基值，故双组优化器必须在每次 step() 前重申比例（policy.DualLrAdam），
往返测试 learn 后的两组 lr 断言即坑防御生效的直接证据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from test_sb3_smoke import N_ABILENE, SEED, STEPS, _make_vec

from trl_sb3.common.config import load_config
from trl_sb3.policy.freeze import apply_freeze, rebuild_optimizer
from trl_sb3.policy.policy import ActorCriticPolicy

TOPO = "Abilene.gml"
LR = 6.0e-6
RATIO = 0.1
CRITIC_LR = 6.0e-7
REINIT_SEED = 42
CRITIC_PREFIX = "mlp_extractor.critic."
HEAD_PREFIX = "mlp_extractor.actor.4."
ACTOR_PREFIX = "mlp_extractor.actor."


def _policy_of(cls: type[ActorCriticPolicy], **policy_kwargs: Any) -> ActorCriticPolicy:
    """独立 policy（Abilene 空间，可传子类）；manual_seed 固定构造期全局 rng 消耗。"""
    vec = _make_vec(TOPO, SEED)
    try:
        th.manual_seed(SEED)
        return cls(vec.observation_space, vec.action_space, lr_schedule=lambda _: LR, **policy_kwargs)
    finally:
        vec.close()


def _policy(**policy_kwargs: Any) -> ActorCriticPolicy:
    return _policy_of(ActorCriticPolicy, **policy_kwargs)


def _dual_ppo(vec: VecEnv, *, ratio: float | None) -> PPO:
    """Abilene 冒烟口径 PPO；policy_kwargs 恒带 lr_critic_ratio 键（runner 同口径，
    None = 缺省单 Adam）。"""
    return PPO(
        policy=ActorCriticPolicy,
        env=vec,
        n_steps=STEPS,
        batch_size=STEPS * N_ABILENE,
        learning_rate=LR,
        policy_kwargs={"lr_critic_ratio": ratio},
        seed=0,
        device="cpu",
    )


def _group_of(policy: ActorCriticPolicy) -> dict[int, int]:
    """id(param) → 组号；跨组重复的参数会让 dict 覆盖——配合前置断言使用。"""
    group_of: dict[int, int] = {}
    for gi, group in enumerate(policy.optimizer.param_groups):
        for p in group["params"]:
            assert id(p) not in group_of, f"参数跨组重复（组 {group_of.get(id(p))} 与 {gi}）"
            group_of[id(p)] = gi
    return group_of


class _PolicyWithStrayParam(ActorCriticPolicy):
    """_build 期注入前缀外参数（模拟未来可学习 features_extractor）。"""

    @override
    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()
        self.stray_param: th.nn.Parameter = th.nn.Parameter(th.zeros(1))


def test_dual_build_two_param_groups() -> None:
    """Given lr_critic_ratio=0.1 的 policy；When 构造（_build）；Then optimizer 是
    Adam 恰两组：actor 组 lr=6e-6、critic 组=6e-7（critic/actor=0.1，DEVIATIONS #3
    legacy 方向）；分区完备互斥——每参数恰落一组，actor 前缀入组 0、critic 入组 1。"""
    policy = _policy(lr_critic_ratio=RATIO)
    assert isinstance(policy.optimizer, th.optim.Adam)
    assert len(policy.optimizer.param_groups) == 2
    assert [g["lr"] for g in policy.optimizer.param_groups] == pytest.approx([LR, CRITIC_LR])
    group_of = _group_of(policy)
    assert set(group_of) == {id(p) for _, p in policy.named_parameters()}, "分区须完备覆盖全部参数"
    for name, p in policy.named_parameters():
        expected = 1 if name.startswith(CRITIC_PREFIX) else 0
        assert group_of[id(p)] == expected, name


def test_null_ratio_single_adam_characterization() -> None:
    """Given 不带 ratio 的 policy + default.yaml 缺省；When 构造；Then 仍是 D3 单
    Adam（type/组数/lr 三重表征，缝 3 逐位不变锚点）且 config 缺省为 null。"""
    policy = _policy()
    assert policy.lr_critic_ratio is None
    assert type(policy.optimizer) is th.optim.Adam
    assert len(policy.optimizer.param_groups) == 1
    assert policy.optimizer.param_groups[0]["lr"] == LR
    assert load_config()["ppo"]["lr_critic_ratio"] is None


def test_dual_lr_ckpt_roundtrip_adapt(tmp_path: Path) -> None:
    """Given 双 lr PPO 先真训两 rollout（优化器有 Adam state）再 save；When
    PPO.load 后真实 adapt 续训一 rollout；Then ratio 属性随 ckpt 往返、learn 后
    仍是恰两组且 lr=[6e-6, 6e-7]（PPO.train 每次 _update_learning_rate 重置组 lr
    的坑被 step 前重申比例挡住）、两组参数都有非平凡 exp_avg state（两组都真实
    步进，非装饰组）。"""
    vec = VecMonitor(_make_vec(TOPO, SEED))
    try:
        model = _dual_ppo(vec, ratio=RATIO)
        model.learn(total_timesteps=2 * STEPS * N_ABILENE)
        path = str(tmp_path / "dual_lr_model")
        model.save(path)
    finally:
        vec.close()
    vec2 = VecMonitor(_make_vec(TOPO, SEED))
    try:
        reloaded = PPO.load(path, env=vec2, device="cpu")
        assert isinstance(reloaded.policy, ActorCriticPolicy)
        assert reloaded.policy.lr_critic_ratio == RATIO
        assert len(reloaded.policy.optimizer.param_groups) == 2  # _setup_model 按存盘 policy_kwargs 重建
        reloaded.learn(total_timesteps=STEPS * N_ABILENE)
        opt = reloaded.policy.optimizer
        assert isinstance(opt, th.optim.Adam)
        assert [g["lr"] for g in opt.param_groups] == pytest.approx([LR, CRITIC_LR])
        for gi in (0, 1):
            group_params = opt.param_groups[gi]["params"]
            assert group_params, "组非空"
            assert all("exp_avg" in opt.state[p] for p in group_params)
            assert any(bool((opt.state[p]["exp_avg"] != 0).any()) for p in group_params)
    finally:
        vec2.close()


def test_freeze_rebuild_dual_mirror() -> None:
    """Given 双 lr PPO + 默认冻结（actor.0-3 冻结、头重初始化可训练）；When
    rebuild_optimizer；Then 仍两组且只含可训练参数：冻结躯干不在任一组、头
    （actor.4）入 actor 组、critic 全入 critic 组，lr=[6e-6, 6e-7]。"""
    vec = _make_vec(TOPO, SEED)
    try:
        model = _dual_ppo(vec, ratio=RATIO)
        # PPO.policy 静态类型是 SB3 基类，运行时必为本类（_dual_ppo 唯一 policy 类）；
        # freeze.py:84 同款 assert 收窄。
        policy = model.policy
        assert isinstance(policy, ActorCriticPolicy)
        apply_freeze(policy, seed=REINIT_SEED)
        rebuild_optimizer(model)
        opt = model.policy.optimizer
        assert isinstance(opt, th.optim.Adam)
        assert [g["lr"] for g in opt.param_groups] == pytest.approx([LR, CRITIC_LR])
        group_of = _group_of(policy)
        for name, p in model.policy.named_parameters():
            if name.startswith(CRITIC_PREFIX):
                assert group_of.get(id(p)) == 1, name
            elif name.startswith(HEAD_PREFIX):
                assert group_of.get(id(p)) == 0, name
            elif name.startswith(ACTOR_PREFIX):
                assert id(p) not in group_of, f"冻结躯干不得入组：{name}"
            else:
                raise AssertionError(f"前缀外参数：{name}")
    finally:
        vec.close()


def test_freeze_rebuild_null_single_group_characterization() -> None:
    """Given 无 ratio 的冻结 PPO；When rebuild_optimizer；Then 单组只含可训练参数、
    lr=6e-6（既有 D3 行为表征——test_freeze.trainable_only 的组数显式补充）。"""
    vec = _make_vec(TOPO, SEED)
    try:
        model = _dual_ppo(vec, ratio=None)
        policy = model.policy
        assert isinstance(policy, ActorCriticPolicy)
        apply_freeze(policy, seed=REINIT_SEED)
        rebuild_optimizer(model)
        opt = model.policy.optimizer
        assert type(opt) is th.optim.Adam
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["lr"] == LR
        opt_ids = {id(p) for p in opt.param_groups[0]["params"]}
        assert opt_ids == {id(p) for p in model.policy.parameters() if p.requires_grad}
    finally:
        vec.close()


def test_dual_build_stray_param_fails_loudly() -> None:
    """Given 构建期注入前缀外参数的 policy（ratio=0.1）；When 构造；Then
    ValueError 点名该参数——分区不完备宁可响亮失败，不静默漏优化。"""
    with pytest.raises(ValueError, match="stray_param"):
        _policy_of(_PolicyWithStrayParam, lr_critic_ratio=RATIO)


def test_rebuild_optimizer_stray_trainable_param_fails_loudly() -> None:
    """Given 双 lr PPO 构建后注入可学习前缀外参数（模拟未来模块）+ 默认冻结；
    When rebuild_optimizer；Then ValueError 点名该参数（镜像路径同口径响亮失败）。"""
    vec = _make_vec(TOPO, SEED)
    try:
        model = _dual_ppo(vec, ratio=RATIO)
        model.policy.stray_param = th.nn.Parameter(th.zeros(1))
        policy = model.policy
        assert isinstance(policy, ActorCriticPolicy)
        apply_freeze(policy, seed=REINIT_SEED)
        with pytest.raises(ValueError, match="stray_param"):
            rebuild_optimizer(model)
    finally:
        vec.close()
