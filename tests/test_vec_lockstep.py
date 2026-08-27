"""NodeFanVecEnv 锁步契约测试（M1-1，决策 D2）：锁步 done / 奖励广播 / 终态 info / reset 原子性 /
确定性 / VecMonitor 冒烟 / get_attr-env_method。

SB3 2.7.0 契约出处（venv 源码核对）：
- done 换算 + terminal_observation + 自动 reset：common/vec_env/dummy_vec_env.py:59-71
- 终态 GAE bootstrap 消费点：common/on_policy_algorithm.py:236-245
  （当且仅当 slot info 同时含 terminal_observation 与 TimeLimit.truncated=True 时
   rewards[idx] += gamma * V(terminal_observation)）
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecMonitor

from trl_sb3.common.config import resolve_path
from trl_sb3.env.node_fan_vec import NodeFanVecEnv
from trl_sb3.env.routing_env import RoutingEnv

SEED = 7
STEPS = 50
N = 11  # Abilene 平行边坍缩后节点数（见 test_env_determinism.test_shapes_and_zero_padding）
OBS_DIM = 7 * 41


def _make_env(seed: int = SEED) -> RoutingEnv:
    return RoutingEnv(resolve_path("topologies/Abilene.gml"), avgrate=500, pbrs=True, seed=seed)


def _make_vec(seed: int = SEED) -> NodeFanVecEnv:
    return NodeFanVecEnv(_make_env(seed))


def _fixed_actions(n: int, steps: int = STEPS) -> np.ndarray:
    """固定动作序列由独立 rng 生成（不依赖 env rng），与 test_env_determinism 同法。"""
    rng = np.random.default_rng(12345)
    return rng.integers(0, 3, size=(steps, n))


def test_lockstep_done() -> None:
    """Given Abilene(11) NodeFanVecEnv；When 50 步固定动作；Then 前 49 步 dones 全 False（且
    TimeLimit.truncated 键每步存在为 False，dummy_vec_env.py:66 同构），第 50 步全 True。"""
    vec = _make_vec()
    vec.reset(seed=SEED)
    actions = _fixed_actions(vec.num_envs)
    for t in range(STEPS - 1):
        _, _, dones, infos = vec.step(actions[t])
        assert dones.shape == (vec.num_envs,) and dones.dtype == np.bool_
        assert not dones.any()
        assert all(info["TimeLimit.truncated"] is False for info in infos)
    _, _, dones, _ = vec.step(actions[-1])
    assert dones.all()


def test_reward_broadcast_equals_underlying_r_mean() -> None:
    """Given vec 与裸 RoutingEnv 同 seed；When 同动作 50 步；Then rewards 每步 N 元素两两相等
    且逐位等于裸 env 的 r_mean（legacy addBufferRD(rewards/node_num) 广播，F13）。"""
    vec = _make_vec()
    vec.reset(seed=SEED)
    ref = _make_env()
    ref.reset(seed=SEED)  # 同为显式重播种，rng 流才对齐（裸 reset() 走构造器续流）
    actions = _fixed_actions(vec.num_envs)
    for t in range(STEPS):
        _, rewards, _, _ = vec.step(actions[t])
        _, r_mean, _, _, _ = ref.step(actions[t])
        assert rewards.shape == (vec.num_envs,)
        assert np.array_equal(rewards, np.full(vec.num_envs, r_mean))


def test_terminal_observation_and_auto_reset() -> None:
    """Given 50 步同动作轨迹（vec 与裸 env 并行重放）；When 第 50 步 truncated；Then 每 slot info
    含 terminal_observation（(287,) 且=本步末观测对应行）与 TimeLimit.truncated=True；本步返回
    obs=新回合 reset 观测（step 不耗 rng、reset 状态不依赖上一回合 → 等于裸 env 第二次 reset）；
    第 51 步继续可用且无 terminal 键。"""
    vec = _make_vec()
    vec.reset(seed=SEED)
    ref = _make_env()
    ref.reset(seed=SEED)  # 同为显式重播种，与 vec 的 rng 流对齐
    actions = _fixed_actions(vec.num_envs)
    for t in range(STEPS):
        final_obs, _, _, truncated, _ = ref.step(actions[t])
        obs, _, dones, infos = vec.step(actions[t])
    assert truncated  # 裸 env 第 50 步确实 truncated（对照前提成立）
    assert dones.all()
    for slot, info in enumerate(infos):
        terminal = info["terminal_observation"]
        assert isinstance(terminal, np.ndarray) and terminal.shape == (OBS_DIM,)
        assert np.array_equal(terminal, final_obs[slot])
        assert info["TimeLimit.truncated"] is True
    new_episode_obs, _ = ref.reset()  # 裸 env 第二次 reset = 自动 reset 应到达的同 rng 流位置
    assert obs.shape == (N, OBS_DIM)
    assert np.array_equal(obs, new_episode_obs)
    obs, _, dones, infos = vec.step(actions[0])  # 第 51 步：新回合第 1 步
    assert not dones.any()
    assert all("terminal_observation" not in info for info in infos)


def test_reset_atomic() -> None:
    """Given vec 与裸 env 同 seed；When vec.reset(seed)；Then 返回 (11,287) 且与裸 env 单次
    reset 逐位相等——若错做 N 次 reset，rng 流推进 N 次，obs 必然不等（原子性证明）。"""
    vec = _make_vec()
    obs = vec.reset(seed=SEED)
    ref = _make_env()
    ref_obs, _ = ref.reset(seed=SEED)  # 同为显式重播种的单次 reset
    assert obs.shape == (N, OBS_DIM) and obs.dtype == np.float64
    assert np.array_equal(obs, ref_obs)


def test_same_seed_vec_bitwise_identical() -> None:
    """Given 同 seed 两 NodeFanVecEnv；When 51 步同动作（含截断与自动 reset）；Then 全程
    obs/rewards/dones 逐位相等。"""
    vec_a, vec_b = _make_vec(), _make_vec()
    obs_a, obs_b = vec_a.reset(seed=SEED), vec_b.reset(seed=SEED)
    assert np.array_equal(obs_a, obs_b)
    actions = _fixed_actions(vec_a.num_envs, steps=STEPS + 1)
    for t in range(STEPS + 1):
        out_a, out_b = vec_a.step(actions[t]), vec_b.step(actions[t])
        assert np.array_equal(out_a[0], out_b[0])
        assert np.array_equal(out_a[1], out_b[1])
        assert np.array_equal(out_a[2], out_b[2])


def test_vec_monitor_smoke() -> None:
    """Given VecMonitor(NodeFanVecEnv)（SB3 Monitor 是 gym Wrapper 包不了 VecEnv，冒烟用
    VecMonitor，vec_monitor.py:75-95 逐 slot 改写 info）；When reset + 51 步；Then 无异常且
    截断步每 slot info 带 episode 统计（r/l/t）。"""
    monitored = VecMonitor(_make_vec())
    monitored.reset()
    actions = _fixed_actions(N, steps=STEPS + 1)
    for t in range(STEPS + 1):
        _, _, dones, infos = monitored.step(actions[t])
        if t == STEPS - 1:
            assert dones.all()
            for info in infos:
                assert {"r", "l", "t"} <= set(info["episode"])


def test_get_attr_env_method_set_attr() -> None:
    """Given vec；When get_attr(n/mu/pbrs) 与 env_method/set_attr；Then 均返回/落到唯一底层的
    N 份相同值。"""
    vec = _make_vec()
    assert vec.num_envs == N
    assert vec.get_attr("n") == [N] * N
    assert vec.get_attr("pbrs") == [True] * N
    mu = vec.get_attr("mu")
    assert len(mu) == N and all(np.array_equal(mu[0], m) for m in mu)
    results = vec.env_method("reset")  # 底层方法返回值（此处为 (obs, info) 元组）复制 N 份
    assert len(results) == N and all(np.array_equal(results[0][0], r[0]) for r in results)
    vec.set_attr("_change_flow_pct", 0.02)
    assert vec.get_attr("_change_flow_pct") == [0.02] * N
