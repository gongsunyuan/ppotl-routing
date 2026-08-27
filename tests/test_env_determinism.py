"""RoutingEnv 确定性与契约测试（规格 J2）：同 seed 逐位相等 / 跨 seed 不同 / 形状零填 /
truncated 时刻 / info 键 / 邻居升序 / k_paths 稳定 / _add_flows 比例缝冒烟。"""

from __future__ import annotations

import numpy as np

from trl_sb3.common.config import resolve_path
from trl_sb3.env.routing_env import RoutingEnv
from trl_sb3.env.topology import k_shortest_paths

TOPOLOGIES = {"CERNET.gml": 41, "Abilene.gml": 11}
INFO_KEYS = {"rd", "rp", "th", "reward_local", "reward_total", "r_mean"}
STEPS = 50


def _make_env(topology: str, pbrs: bool = True, seed: int = 7) -> RoutingEnv:
    return RoutingEnv(resolve_path(f"topologies/{topology}"), avgrate=500, pbrs=pbrs, seed=seed)


def _fixed_actions(n: int) -> np.ndarray:
    """固定动作序列由独立 rng 生成（不依赖 env rng）。"""
    rng = np.random.default_rng(12345)
    return rng.integers(0, 3, size=(STEPS, n))


def _assert_info_equal(info_a: dict, info_b: dict) -> None:
    """info 键齐全且逐位相等（数组含 dtype）。"""
    assert set(info_a) == INFO_KEYS
    for key in sorted(INFO_KEYS):
        va, vb = info_a[key], info_b[key]
        if isinstance(va, np.ndarray):
            assert np.array_equal(va, vb) and va.dtype == vb.dtype
        else:
            assert va == vb


def test_same_seed_bitwise_identical_trajectory() -> None:
    """Given 同 seed 两实例（pbrs on/off 各跑）；When 50 步固定动作全轨迹；Then obs/reward/info 逐位相等。"""
    for pbrs in (True, False):
        for topo, n in TOPOLOGIES.items():
            actions = _fixed_actions(n)
            env_a = _make_env(topo, pbrs=pbrs, seed=7)
            env_b = _make_env(topo, pbrs=pbrs, seed=7)
            obs_a, info_a = env_a.reset()
            obs_b, info_b = env_b.reset()
            assert np.array_equal(obs_a, obs_b) and obs_a.dtype == obs_b.dtype
            _assert_info_equal(info_a, info_b)
            for t in range(STEPS):
                out_a = env_a.step(actions[t])
                out_b = env_b.step(actions[t])
                assert np.array_equal(out_a[0], out_b[0]) and out_a[0].dtype == out_b[0].dtype
                assert out_a[1] == out_b[1]
                assert out_a[2] is False  # terminated 恒 False
                assert out_a[3] == (t == STEPS - 1)  # truncated 恰在第 50 步
                _assert_info_equal(out_a[4], out_b[4])


def test_cross_seed_differs() -> None:
    """Given 不同 seed 两实例；When reset+一步；Then 轨迹不同。"""
    actions = _fixed_actions(41)
    env_a = _make_env("CERNET.gml", seed=7)
    env_b = _make_env("CERNET.gml", seed=8)
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    assert not np.array_equal(obs_a, obs_b)
    step_a = env_a.step(actions[0])
    step_b = env_b.step(actions[0])
    assert not np.array_equal(step_a[0], step_b[0]) or step_a[1] != step_b[1]


def test_shapes_and_zero_padding() -> None:
    """Given CERNET/Abilene；When reset；Then (41,287)/(11,287) 且 Abilene 行尾 210 个零。"""
    env_cernet = _make_env("CERNET.gml")
    obs_cernet, _ = env_cernet.reset()
    assert obs_cernet.shape == (41, 287) and obs_cernet.dtype == np.float64
    env_abilene = _make_env("Abilene.gml")
    obs_abilene, _ = env_abilene.reset()
    assert obs_abilene.shape == (11, 287)
    assert env_cernet._n == 41 and env_abilene._n == 11  # N§3：平行边坍缩后节点数
    assert np.all(obs_abilene[:, 7 * 11 :] == 0.0)  # 7*(41-11)=210 个零


def test_neighbors_ascending() -> None:
    """Given 环境；When 取邻居表；Then 每节点邻居严格升序（D7 稳定化）。"""
    for topo in TOPOLOGIES:
        env = _make_env(topo)
        for nbrs in env._neighbors:
            assert nbrs == sorted(nbrs)


def test_k_paths_deterministic_and_cached() -> None:
    """Given 同一赋权图；When 两次 k_shortest_paths / 两次 _paths_for；Then 结果相等且缓存命中。"""
    env = _make_env("CERNET.gml")
    env.reset()
    paths_a = [k_shortest_paths(env._graph, i, 5, 3) for i in (0, 10, 20, 30)]
    paths_b = [k_shortest_paths(env._graph, i, 5, 3) for i in (0, 10, 20, 30)]
    assert paths_a == paths_b
    dst0 = int(env._dst[0])
    assert env._paths_for(0, dst0) is env._paths_for(0, dst0)


def test_add_flows_proportion_seam() -> None:
    """Given 非 one-hot 手工比例 split（分流缝）；When _add_flows+_get_reward；Then 可跑且全有限。"""
    env = _make_env("Abilene.gml")
    env.reset()
    split = np.zeros((env._n, 3))
    for i in range(env._n):
        for k in range(len(env._flow_paths[i])):
            split[i, k] = (0.5, 0.3, 0.2)[k]
    env._add_flows(split)
    reward = env._get_reward(split)
    assert np.all(np.isfinite(reward.local)) and np.all(np.isfinite(reward.total))
