"""M1-2 policy 冒烟测试（决策 D3）：nets 结构/kaiming 复现性、SB3 PPO 接线与 learn、
ckpt 保存/加载逐位相等、跨拓扑 ckpt 可移植（F2/D1）、VecMonitor 回合统计不放大（D2 除 N）、
deterministic predict 出合法离散动作。

SB3 2.7.0 接线出处（venv 源码核对）：common/policies.py——_build :585-634、
_build_mlp_extractor :570-583、forward :636-658、evaluate_actions :719-741、
get_distribution :743-752、predict_values :754-763、ortho_init 块 :610-631。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

from trl_sb3.common.config import resolve_path
from trl_sb3.env.node_fan_vec import NodeFanVecEnv
from trl_sb3.env.routing_env import RoutingEnv
from trl_sb3.policy.nets import build_nets
from trl_sb3.policy.policy import ActorCriticPolicy

SEED = 7
PPO_SEED = 0
N_ABILENE = 11
N_CERNET = 41
OBS_DIM = 287
STEPS = 50


def _make_vec(topo: str, seed: int) -> NodeFanVecEnv:
    env = RoutingEnv(resolve_path(f"topologies/{topo}"), avgrate=500, pbrs=True, seed=seed)
    return NodeFanVecEnv(env)


def _ppo(vec: VecEnv, *, n_steps: int = 50, batch_size: int = 550, n_epochs: int = 10) -> PPO:
    """冒烟超参在测试内显式传（论文口径见 config/default.yaml ppo 节）。"""
    return PPO(
        policy=ActorCriticPolicy,
        env=vec,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=6.0e-6,
        gamma=0.99,
        gae_lambda=1.0,
        clip_range=0.2,
        ent_coef=0.05,
        normalize_advantage=True,
        verbose=0,
        seed=PPO_SEED,
        device="cpu",
    )


def _fixed_actions(n: int, steps: int = STEPS) -> npt.NDArray[np.int64]:
    rng = np.random.default_rng(12345)
    return rng.integers(0, 3, size=(steps, n))


def _layer_shapes(net: th.nn.Module) -> list[tuple[int, ...]]:
    return [tuple(p.shape) for p in net.parameters() if p.ndim == 2]


def test_build_nets_structure_and_kaiming_reproducibility() -> None:
    """Given build_nets(287, 3, seed)；When 构网两次；Then actor/critic 逐层 shape 镜像
    legacy main.py:81-111（287→128→64→3 / →1、PReLU、无 Tanh），同 seed 两次 state_dict
    逐位相等、异 seed 权重不同（kaiming 由 torch.Generator 控制，F2/D3）。"""
    actor, critic = build_nets(obs_dim=OBS_DIM, n_actions=3, seed=SEED)
    assert [type(m) for m in actor] == [th.nn.Linear, th.nn.PReLU, th.nn.Linear, th.nn.PReLU, th.nn.Linear]
    assert _layer_shapes(actor) == [(128, 287), (64, 128), (3, 64)]
    assert [type(m) for m in critic][:4] == [th.nn.Linear, th.nn.PReLU, th.nn.Linear, th.nn.PReLU]
    assert _layer_shapes(critic) == [(128, 287), (64, 128), (1, 64)]
    assert not any(isinstance(m, th.nn.Tanh) for m in (*actor, *critic))
    actor_b, critic_b = build_nets(obs_dim=OBS_DIM, n_actions=3, seed=SEED)
    for net_a, net_b in ((actor, actor_b), (critic, critic_b)):
        for pa, pb in zip(net_a.parameters(), net_b.parameters(), strict=True):
            assert th.equal(pa, pb)
    actor_c, _ = build_nets(obs_dim=OBS_DIM, n_actions=3, seed=SEED + 1)
    assert not th.equal(next(actor.parameters()), next(actor_c.parameters()))


def test_policy_wiring_bypasses_mlp_extractor() -> None:
    """Given 直接构造 ActorCriticPolicy（Abilene 空间）；When 构建完成；Then pi/vf 为独立
    两网（share_features_extractor=False、无 MlpExtractor、无 Tanh），action_net/value_net
    无参数（Identity），float64 obs forward 出有限动作与值（preprocess_obs .float() 铸型）。"""
    vec = _make_vec("Abilene.gml", SEED)
    policy = ActorCriticPolicy(vec.observation_space, vec.action_space, lr_schedule=lambda _: 6.0e-6)
    assert policy.share_features_extractor is False
    assert not isinstance(policy.mlp_extractor, MlpExtractor)
    assert _layer_shapes(policy.mlp_extractor.actor) == [(128, 287), (64, 128), (3, 64)]
    assert _layer_shapes(policy.mlp_extractor.critic) == [(128, 287), (64, 128), (1, 64)]
    assert sum(p.numel() for p in policy.action_net.parameters()) == 0
    assert sum(p.numel() for p in policy.value_net.parameters()) == 0
    assert not any(isinstance(m, th.nn.Tanh) for m in policy.modules())
    obs = vec.reset()
    actions, values, log_prob = policy(th.as_tensor(obs))
    assert actions.shape == (N_ABILENE,) and th.isfinite(actions).all()
    assert values.shape == (N_ABILENE, 1) and th.isfinite(values).all()
    assert th.isfinite(log_prob).all()
    vec.close()


def test_ppo_learn_smoke_cpu() -> None:
    """Given Abilene NodeFanVecEnv+VecMonitor 上的 PPO（论文口径超参）；When learn(50*11*4)；
    Then 无异常且 evaluate_actions 输出 values/log_prob/entropy 全有限（无 NaN/Inf）。"""
    vec = VecMonitor(_make_vec("Abilene.gml", SEED))
    model = _ppo(vec)
    model.learn(total_timesteps=STEPS * N_ABILENE * 4)
    obs = th.as_tensor(vec.reset())
    actions = th.zeros(N_ABILENE, dtype=th.int64)
    values, log_prob, entropy = model.policy.evaluate_actions(obs, actions)
    for tensor in (values, log_prob, entropy):  # entropy: Tensor | None（evaluate_actions 口径）
        assert tensor is not None
        assert th.isfinite(tensor).all()
    vec.close()


def test_save_load_roundtrip_bitwise(tmp_path: Any) -> None:
    """Given learn 一个 rollout 后的 PPO；When save→PPO.load；Then policy 参数逐位相等，
    产物落在 tmp_path（pytest 自动清理）。"""
    vec = VecMonitor(_make_vec("Abilene.gml", SEED))
    model = _ppo(vec)
    model.learn(total_timesteps=STEPS * N_ABILENE)
    path = str(tmp_path / "model")
    model.save(path)
    reloaded = PPO.load(path, device="cpu")
    for pa, pb in zip(model.policy.parameters(), reloaded.policy.parameters(), strict=True):
        assert th.equal(pa, pb)
    vec.close()


def test_cross_topology_ckpt_portable(tmp_path: Any) -> None:
    """Given CERNET(41) 上训几步保存的 ckpt；When PPO.load 后在 Abilene NodeFanVecEnv 上
    predict；Then 输出 (11,) 合法离散动作——SB3 ckpt 只依赖 obs 形状 287（F2/D1 可移植根基）。"""
    vec_cernet = VecMonitor(_make_vec("CERNET.gml", SEED))
    model = _ppo(vec_cernet, n_steps=10, batch_size=10 * N_CERNET, n_epochs=2)
    model.learn(total_timesteps=10 * N_CERNET)
    path = str(tmp_path / "cernet_model")
    model.save(path)
    del model
    vec_abilene = _make_vec("Abilene.gml", SEED)
    reloaded = PPO.load(path, device="cpu")
    obs = vec_abilene.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (N_ABILENE, OBS_DIM)
    actions, _ = reloaded.predict(obs, deterministic=True)
    assert actions.shape == (N_ABILENE,)
    assert actions.dtype.kind == "i"
    assert ((0 <= actions) & (actions <= 2)).all()
    vec_cernet.close()
    vec_abilene.close()


def test_vec_monitor_episode_reward_not_amplified() -> None:
    """Given VecMonitor(NodeFanVecEnv(Abilene))；When 50 步固定动作跑满一回合；Then 每 slot
    info["episode"]["r"] ≈ Σ本回合 r_mean（float32 累加容差内）——若奖励未除 N 会放大 11 倍
    即刻暴露（D2 关注点实测锁定）。"""
    vec = VecMonitor(_make_vec("Abilene.gml", SEED))
    vec.reset()
    actions = _fixed_actions(N_ABILENE)
    r_mean_sum = 0.0
    infos: list[dict[str, Any]] = []
    for t in range(STEPS):
        _, rewards, dones, infos = vec.step(actions[t])
        r_mean_sum += float(rewards[0])
        assert dones.all() == (t == STEPS - 1)
    assert abs(r_mean_sum) > 1.0  # 对照前提：奖励量级足以区分 11 倍放大
    for info in infos:
        episode: dict[str, float] = info["episode"]
        assert episode["l"] == STEPS
        assert episode["r"] == pytest.approx(r_mean_sum, rel=1e-4, abs=1e-6)
    vec.close()


def test_predict_deterministic_discrete_actions() -> None:
    """Given Abilene 上的 PPO（未训练）；When predict(deterministic=True)；Then (11,) int
    动作且取值 ∈ {0,1,2}。"""
    vec = _make_vec("Abilene.gml", SEED)
    model = _ppo(vec)
    obs = vec.reset()
    assert isinstance(obs, np.ndarray)
    actions, _ = model.predict(obs, deterministic=True)
    assert actions.shape == (N_ABILENE,)
    assert actions.dtype.kind == "i"
    assert ((0 <= actions) & (actions <= 2)).all()
    vec.close()
