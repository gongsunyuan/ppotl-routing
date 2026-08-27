"""PBRS 伸缩恒等式与 shaped/unshaped 回报等价测试（计划 §7 第 2 条 / M2-2）。

数学依据（routing_env._get_reward 已实现形式，G_t = beta_delay·rd_t + alpha_plr·rp_t，G_0 = Φ(s_0) = 0）：
  shaped_t = (1+γ)·G_t − G_{t−1}
  ⇒  Σ_{t=1..T} γ^{t−1}·shaped_t = Σ_{t=1..T} γ^{t−1}·G_t + γ^T·G_T    （telescoping；pbrs=False 时边界项为 0）
即固定轨迹上 shaped 回报 = unshaped 回报 + 边界势差 γ^T·G_T。

cross-mode 前提说明：change_flows 消费含 PBRS 项的 total（legacy routingEnv.py:366→473 同构），
pbrs=True/False 同 seed 同动作的轨迹自第 2 步起分歧（非移植 bug）→ 回报等价按计划 §7-2 的
"固定轨迹上"口径在单一 pbrs=True 轨迹验证；数值与根因见 notepads issues.md M2-2 条目。
"""

from __future__ import annotations

import numpy as np
import pytest

from trl_sb3.common.config import resolve_path
from trl_sb3.env.routing_env import RoutingEnv

TOPOLOGIES = {"CERNET.gml": 41, "Abilene.gml": 11}
STEPS = 50
RTOL, ATOL = 1e-9, 1e-12


def _make_env(topology: str, pbrs: bool, seed: int = 7) -> RoutingEnv:
    return RoutingEnv(resolve_path(f"topologies/{topology}"), avgrate=500, pbrs=pbrs, seed=seed)


def _fixed_actions(n: int) -> np.ndarray:
    """固定动作序列由独立 rng 生成（不依赖 env rng，沿用 test_env_determinism 写法）。"""
    rng = np.random.default_rng(12345)
    return rng.integers(0, 3, size=(STEPS, n))


def _rollout(
    env: RoutingEnv, actions: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """reset 后跑满 STEPS 步，返回 (obs 序列, reward_local 序列, G 序列)；G 由 info 的 rd/rp 现算。"""
    beta, alpha = env._beta_delay, env._alpha_plr
    env.reset()
    obs_seq: list[np.ndarray] = []
    local_seq: list[np.ndarray] = []
    g_seq: list[np.ndarray] = []
    for t in range(STEPS):
        obs, _, _, _, info = env.step(actions[t])
        obs_seq.append(obs)
        local_seq.append(info["reward_local"])
        g_seq.append(beta * info["rd"] + alpha * info["rp"])
    return obs_seq, local_seq, g_seq


@pytest.mark.parametrize("pbrs", [True, False], ids=["pbrs-on", "pbrs-off"])
@pytest.mark.parametrize("topology", TOPOLOGIES, ids=list(TOPOLOGIES))
def test_telescoping_identity(topology: str, pbrs: bool) -> None:
    """Given pbrs 任一模的 50 步固定动作真实轨迹；When 按第 t 步折扣 γ^{t-1} 累加 reward_local；
    Then per-node 恒等于 Σγ^{t-1}·G_t + γ^50·G_50（pbrs=False 边界项为 0，即 unshaped 回报=base 回报）。"""
    env = _make_env(topology, pbrs=pbrs)
    _, local_seq, g_seq = _rollout(env, _fixed_actions(TOPOLOGIES[topology]))
    gamma = env._gamma
    disc = gamma ** np.arange(STEPS)  # 第 t 步（1 起）折扣 γ^{t-1}
    shaped_return = disc @ np.array(local_seq)
    base_return = disc @ np.array(g_seq)
    boundary = (gamma**STEPS) * g_seq[-1] if pbrs else np.zeros(TOPOLOGIES[topology])
    assert np.allclose(shaped_return, base_return + boundary, rtol=RTOL, atol=ATOL), (
        f"{topology} pbrs={pbrs}: max|lhs-rhs|={np.abs(shaped_return - base_return - boundary).max():.3e}"
    )


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=list(TOPOLOGIES))
def test_shaped_unshaped_return_equivalence(topology: str) -> None:
    """Given pbrs=True 单一固定轨迹（unshaped 回报由同一轨迹的 G_t 现算）；
    When shaped/unshaped 折扣回报作差；Then per-node 与节点均值（r_mean 口径）均 == γ^50·G_50。
    均值口径用 reward_local：env 的 r_mean 源自 round(,2) 的 total，取整破坏恒等式故不作断言。"""
    env = _make_env(topology, pbrs=True)
    _, local_seq, g_seq = _rollout(env, _fixed_actions(TOPOLOGIES[topology]))
    gamma = env._gamma
    disc = gamma ** np.arange(STEPS)
    shaped_return = disc @ np.array(local_seq)
    unshaped_return = disc @ np.array(g_seq)
    boundary = (gamma**STEPS) * g_seq[-1]
    assert np.allclose(shaped_return - unshaped_return, boundary, rtol=RTOL, atol=ATOL), (
        f"{topology}: per-node max|Δ-boundary|="
        f"{np.abs(shaped_return - unshaped_return - boundary).max():.3e}"
    )
    assert np.allclose(
        shaped_return.mean() - unshaped_return.mean(), boundary.mean(), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=list(TOPOLOGIES))
def test_first_step_shaped_form(topology: str) -> None:
    """Given pbrs=True 且刚 reset（G_prev = Φ(s_0) = 0）；When 第 1 步；Then reward_local == G_1·(1+γ)。"""
    env = _make_env(topology, pbrs=True)
    _, local_seq, g_seq = _rollout(env, _fixed_actions(TOPOLOGIES[topology]))
    assert np.allclose(local_seq[0], (1.0 + env._gamma) * g_seq[0], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=list(TOPOLOGIES))
def test_cross_mode_first_step_shared_but_dynamics_diverge(topology: str) -> None:
    """Given 同 seed 同固定动作的 pbrs=True/False 两实例；When 各跑 50 步；
    Then 第 1 步 G 逐位相同（结构性：change_flows 在奖励计算后才改 rates），
    但完整 obs 序列存在分歧——shaped total 经 change_flows 反馈进速率动力学
    （legacy routingEnv.py:366→473 同构，legacy-faithful 而非移植 bug）。
    本测试锁定该行为：回报等价因此只能在固定轨迹上验证（见上测试与 issues.md M2-2）。"""
    actions = _fixed_actions(TOPOLOGIES[topology])
    obs_on, _, g_on = _rollout(_make_env(topology, pbrs=True), actions)
    obs_off, _, g_off = _rollout(_make_env(topology, pbrs=False), actions)
    assert np.array_equal(g_on[0], g_off[0])
    diverged = [t for t in range(STEPS) if not np.array_equal(obs_on[t], obs_off[t])]
    assert diverged, (
        f"{topology}: pbrs on/off 轨迹意外逐位相同——若 change_flows 已改用 base reward，"
        "请把回报等价测试改回双实例对照并更新 issues.md M2-2"
    )
