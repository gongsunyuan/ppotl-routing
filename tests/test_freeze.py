"""M2-1 冻结语义测试（决策 D4）：apply_freeze/rebuild_optimizer 契约。

覆盖：requires_grad 状态（config frozen_layers=actor.0-3 含两 PReLU 口径）、
冻结参数训练全程零梯度零漂移、头重初始化范数变化与躯干逐位不变、
seed 复现、优化器只含可训练参数、幂等。

SB3 2.7.0 出处：ppo/ppo.py:274-278 走 loss.backward()+optimizer.step()，
requires_grad=False 参数不建计算图 → .grad 恒 None；train 末 zero_grad()
（set_to_none=True）→ "在学"断言用参数值变化而非残留 grad。
"""

from __future__ import annotations

import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from test_sb3_smoke import SEED, _make_vec, _ppo

from trl_sb3.policy.freeze import apply_freeze, rebuild_optimizer
from trl_sb3.policy.policy import ActorCriticPolicy

REINIT_SEED = 42
N_ABILENE = 11
STEPS = 50
HEAD_PREFIX = "mlp_extractor.actor.4."


def _policy() -> ActorCriticPolicy:
    """独立 policy（Abilene 空间）；manual_seed 使构造期全局 rng 消耗可复现。"""
    vec = _make_vec("Abilene.gml", SEED)
    try:
        th.manual_seed(SEED)
        return ActorCriticPolicy(vec.observation_space, vec.action_space, lr_schedule=lambda _: 6.0e-6)
    finally:
        vec.close()


def _head(net: th.nn.Sequential) -> th.nn.Linear:
    """net[-1] 静态类型是 Sequential | Module（torch stub）；收窄到输出头 Linear。"""
    head = net[-1]
    assert isinstance(head, th.nn.Linear)
    return head


def test_apply_freeze_requires_grad_states() -> None:
    """Given Abilene policy；When apply_freeze；Then 冻结清单 actor.0/1/2/3（含两
    PReLU，D4 除输出头外全冻结）全 False，头 actor.4 True，critic 全 True。"""
    policy = _policy()
    apply_freeze(policy, seed=REINIT_SEED)
    actor = policy.mlp_extractor.actor
    for idx in (0, 1, 2, 3):
        assert all(not p.requires_grad for p in actor[idx].parameters())
    assert all(p.requires_grad for p in actor[-1].parameters())
    assert all(p.requires_grad for p in policy.mlp_extractor.critic.parameters())


def test_frozen_params_zero_grad_and_unchanged_through_training() -> None:
    """Given 冻结 PPO（Abilene 冒烟口径）；When learn 2 个 rollout；Then 冻结参数
    .grad None/全零且逐位不变，头与 critic 参数值变化（真的在学，train 末 zero_grad
    后 grad 不可残留断言，以参数位移证学习）。"""
    vec = VecMonitor(_make_vec("Abilene.gml", SEED))
    model = _ppo(vec)
    assert isinstance(model.policy, ActorCriticPolicy)
    apply_freeze(model.policy, seed=REINIT_SEED)
    rebuild_optimizer(model)
    before = {n: p.detach().clone() for n, p in model.policy.named_parameters()}
    assert any(not p.requires_grad for p in model.policy.parameters())  # 前提：确有冻结
    model.learn(total_timesteps=2 * STEPS * N_ABILENE)
    changed: list[str] = []
    for name, param in model.policy.named_parameters():
        if not param.requires_grad:
            assert param.grad is None or bool((param.grad == 0).all())
            assert th.equal(before[name], param.detach())
        elif not th.equal(before[name], param.detach()):
            changed.append(name)
    assert any(n.startswith(HEAD_PREFIX) for n in changed)
    assert any(n.startswith("mlp_extractor.critic.") for n in changed)
    vec.close()


def test_reinit_head_changes_norm_torso_bitwise_unchanged() -> None:
    """Given apply_freeze 前参数快照；When apply_freeze(seed)；Then 头 weight/bias
    not torch.equal 且范数差>0，躯干与 critic 全部参数逐位不变（仅头被重置）。"""
    policy = _policy()
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    apply_freeze(policy, seed=REINIT_SEED)
    head_w = _head(policy.mlp_extractor.actor).weight.detach()
    head_b = _head(policy.mlp_extractor.actor).bias.detach()
    assert not th.equal(before["mlp_extractor.actor.4.weight"], head_w)
    assert not th.equal(before["mlp_extractor.actor.4.bias"], head_b)
    assert abs(before["mlp_extractor.actor.4.weight"].norm().item() - head_w.norm().item()) > 0.0
    for name, param in policy.named_parameters():
        if not name.startswith(HEAD_PREFIX):
            assert th.equal(before[name], param.detach())


def test_reinit_seed_reproducible() -> None:
    """Given 同一 policy；When 同 seed 两次 apply_freeze 与异 seed 一次；Then 同 seed
    头参数逐位相等，异 seed 不同（Generator 控制，PPO 全局 rng 不受影响）。"""
    policy = _policy()
    apply_freeze(policy, seed=REINIT_SEED)
    head = _head(policy.mlp_extractor.actor)
    w_first = head.weight.detach().clone()
    b_first = head.bias.detach().clone()
    apply_freeze(policy, seed=REINIT_SEED)
    assert th.equal(w_first, head.weight.detach())
    assert th.equal(b_first, head.bias.detach())
    apply_freeze(policy, seed=REINIT_SEED + 1)
    assert not th.equal(w_first, head.weight.detach())


def test_rebuild_optimizer_trainable_only() -> None:
    """Given 冻结后的 PPO；When rebuild_optimizer；Then optimizer 参数集 == requires_grad
    参数集（冻结参数无 optimizer 状态），单 Adam lr=6.0e-6（D3，读 model.learning_rate）。"""
    vec = _make_vec("Abilene.gml", SEED)
    model = PPO(
        policy=ActorCriticPolicy,
        env=vec,
        n_steps=50,
        batch_size=50 * N_ABILENE,
        learning_rate=6.0e-6,
        seed=0,
        device="cpu",
    )
    assert isinstance(model.policy, ActorCriticPolicy)
    apply_freeze(model.policy, seed=REINIT_SEED)
    rebuild_optimizer(model)
    opt_ids = {id(p) for group in model.policy.optimizer.param_groups for p in group["params"]}
    trainable_ids = {id(p) for p in model.policy.parameters() if p.requires_grad}
    frozen_ids = {id(p) for p in model.policy.parameters() if not p.requires_grad}
    assert opt_ids == trainable_ids
    assert not opt_ids & frozen_ids
    assert model.policy.optimizer.param_groups[0]["lr"] == 6.0e-6
    assert type(model.policy.optimizer) is th.optim.Adam
    vec.close()


def test_apply_freeze_idempotent_requires_grad() -> None:
    """Given apply_freeze 一次后的 requires_grad 快照；When 再调用一次；Then 全参数
    requires_grad 状态逐项不变（幂等；头参数另行被同 seed 重置为相同值）。"""
    policy = _policy()
    apply_freeze(policy, seed=REINIT_SEED)
    snapshot = {n: p.requires_grad for n, p in policy.named_parameters()}
    apply_freeze(policy, seed=REINIT_SEED)
    assert {n: p.requires_grad for n, p in policy.named_parameters()} == snapshot


@pytest.mark.skipif(not th.cuda.is_available(), reason="needs cuda")
def test_apply_freeze_cuda_roundtrip() -> None:
    """Given device='cuda' 的 Abilene 冒烟 PPO；When apply_freeze(seed)+rebuild_optimizer
    +learn(50*11)；Then 全程无异常、头参数有限（回归：CPU Generator 对 CUDA 张量
    kaiming 抛 RuntimeError——Generator 必须随头参数设备创建）。"""
    vec = VecMonitor(_make_vec("Abilene.gml", SEED))
    model = PPO(
        policy=ActorCriticPolicy,
        env=vec,
        n_steps=50,
        batch_size=50 * N_ABILENE,
        n_epochs=10,
        learning_rate=6.0e-6,
        verbose=0,
        seed=0,
        device="cuda",
    )
    assert isinstance(model.policy, ActorCriticPolicy)
    apply_freeze(model.policy, seed=99)
    rebuild_optimizer(model)
    model.learn(total_timesteps=STEPS * N_ABILENE)
    head = _head(model.policy.mlp_extractor.actor)
    assert th.isfinite(head.weight.detach().cpu()).all()
    assert th.isfinite(head.bias.detach().cpu()).all()
    vec.close()
