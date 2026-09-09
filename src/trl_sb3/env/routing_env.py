"""Gymnasium 版 RoutingEnv：legacy TRL_Routing/code/routingEnv.py 的 clean-room 移植（规格 B-H）。

legacy 行号出处：
- 拓扑/直径/邻居/μ/整数边权：routingEnv.py:28-65, 106-108（经 env.topology）
- zip_dist 候选速率与 [0.6,0.3,0.1]：routingEnv.py:112-132（第三轮定案，N§1）
- reset / flow 生成 / 速率扰动：routingEnv.py:153-208
- add_flows 传播（10 轮、节点升序、out_rate 首录胜出）：routingEnv.py:210-266
- get_reward（μ=源节点 quirk、PBRS、0.6/0.4 混合、round(,2)）：routingEnv.py:267-371
- get_state 7 维观测与邻居零填：routingEnv.py:372-423（rate/avg_rate:385、th/2000:301）
- step / change_flows（stable argsort、i<=N/2、±1%）：routingEnv.py:452-481
与 legacy 的差异记 .omo notepads issues.md（PBRS Φ(s0)=0、out_rate 缺省 0.0、th 先算后存）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, override

import gymnasium as gym
import networkx as nx
import numpy as np
import numpy.typing as npt

from trl_sb3.env.mmk import mmk_delay, mmk_loss
from trl_sb3.env.topology import k_shortest_paths, load_topology

if TYPE_CHECKING:
    from trl_sb3.env.topology import _DiGraph

# N§1 定案：zip_dist 的 Poisson 权重 [1, 1/2, 1/6] 归一化 → [0.6, 0.3, 0.1]。
ZIP_PROB: tuple[float, float, float] = (0.6, 0.3, 0.1)


class _StepReward(NamedTuple):
    """_get_reward 输出：每节点 local/total 与本步 r_d/r_p，及均值 r_mean。"""

    local: npt.NDArray[np.float64]
    total: npt.NDArray[np.float64]
    r_mean: float
    rd: npt.NDArray[np.float64]
    rp: npt.NDArray[np.float64]


class RoutingEnv(gym.Env[npt.NDArray[np.float64], np.int64]):
    """节点扇出多智能体路由环境：每节点 7*max_nodes 维观测、Discrete(n_candidates) 动作。

    metadata 不覆写：与 gym.Env 默认 {"render_modes": []} 完全一致，直接继承。
    ActType=np.int64（per-node Discrete 口径）；D2 节点扇出下 step 实收全节点
    动作向量 (N,)——形参取 int | 向量 并集以同时满足 gym 覆写与扇出调用方。"""

    def __init__(
        self,
        gml_path: str | Path,
        avgrate: float,
        alpha_plr: float = 0.7,
        beta_delay: float = 0.3,
        *,
        pbrs: bool,
        seed: int,
        max_nodes: int = 41,
        n_candidates: int = 3,
        capacity: int = 10000,
        mu_choices: tuple[int, ...] = (1000, 2000, 3000),
        gamma: float = 0.99,
        mix_local: float = 0.6,
        mix_global: float = 0.4,
        episode_steps: int = 50,
        th_max_rate: float = 2000.0,
        zip_interval: int = 100,
        rate_noise_scale: float = 10.0,
        rate_noise_clip: float = 20.0,
        propagation_rounds: int = 10,
        change_flow_pct: float = 0.01,
        change_flow_by_ep: tuple[tuple[int, float], ...] = ((8000, 1.0),),
    ) -> None:
        super().__init__()
        self._pbrs: bool = pbrs
        self._alpha_plr: float = alpha_plr
        self._beta_delay: float = beta_delay
        self._n_candidates: int = n_candidates
        self._capacity: int = capacity
        self._gamma: float = gamma
        self._mix_local: float = mix_local
        self._mix_global: float = mix_global
        self._episode_steps: int = episode_steps
        self._th_max_rate: float = th_max_rate
        self._rate_noise_scale: float = rate_noise_scale
        self._rate_noise_clip: float = rate_noise_clip
        self._propagation_rounds: int = propagation_rounds
        self._change_flow_pct: float = change_flow_pct
        self._change_flow_by_ep: tuple[tuple[int, float], ...] = change_flow_by_ep
        self._avgrate: float = float(avgrate)
        self._max_nodes: int = max_nodes
        self._rng: np.random.Generator = np.random.default_rng(
            seed
        )  # 单流 rng，不碰全局
        self._graph: _DiGraph = load_topology(gml_path)
        self._n: int = self._graph.number_of_nodes()
        self._max_hop: int = nx.diameter(self._graph)
        self._mu: npt.NDArray[np.int64] = self._rng.choice(
            mu_choices, size=self._n
        )  # ① init 一次性，此后图静态
        lcm = math.lcm(*mu_choices)
        for u, v in self._graph.edges:
            self._graph[u][v]["weight"] = lcm // int(self._mu[v])  # 覆盖矩阵残留值
        self._neighbors: list[list[int]] = [
            sorted(self._graph.neighbors(i)) for i in range(self._n)
        ]
        base_rate = int(avgrate)
        self._candidate_rates: npt.NDArray[np.int64] = np.array(
            [base_rate, base_rate + zip_interval, base_rate - zip_interval]
        )
        self._dst: npt.NDArray[np.int64] = np.empty(self._n, dtype=np.int64)
        self._rate_init: npt.NDArray[np.int64] = np.empty(self._n, dtype=np.int64)
        for i in range(self._n):  # ② 每 flow：dst（重抽至 ≠i）→ rate_init
            dst = int(self._rng.integers(0, self._n))
            while dst == i:
                dst = int(self._rng.integers(0, self._n))
            self._dst[i] = dst
            self._rate_init[i] = int(
                self._rng.choice(self._candidate_rates, p=list(ZIP_PROB))
            )
        # 基类 gym.Env 已声明 Space[ObsType]/Space[ActType]——按基类口径注解（覆写
        # 兼容），具体 Box/Discrete 由消费方 isinstance 收窄（policy.py:173 同法）。
        self.observation_space: gym.spaces.Space[npt.NDArray[np.float64]] = (
            gym.spaces.Box(-np.inf, np.inf, (78,), dtype=np.float64)
        )
        self.action_space: gym.spaces.Space[np.int64] = gym.spaces.Discrete(
            n_candidates
        )
        self._t: int = 0
        self._episode: int = 0
        self._rates: npt.NDArray[np.float64] = np.zeros(self._n)
        self._last_rd: npt.NDArray[np.float64] = np.zeros(self._n)
        self._last_rp: npt.NDArray[np.float64] = np.zeros(self._n)
        self._k_path_cache: dict[tuple[int, int], list[list[int]]] = {}
        self._flow_paths: list[list[list[int]]] | None = None
        self._in_rate: list[dict[int, float]] = []
        self._out_rate: list[dict[int, float]] = []

    def _paths_for(self, src: int, dst: int) -> list[list[int]]:
        """实例级 {(src,dst): paths} 缓存（src/dst init 后不变 → 图静态，规格 A）。"""
        cached = self._k_path_cache.get((src, dst))
        if cached is None:
            cached = k_shortest_paths(self._graph, src, dst, self._n_candidates)
            self._k_path_cache[(src, dst)] = cached
        return cached

    def _paths(self) -> list[list[list[int]]]:
        """_flow_paths 的非 None 视图：k_paths 在首次 reset() 惰性初始化（docstring 见 reset），
        本方法及全部消费方（step 链/启发式 seam）只在 reset 之后调用——gym 契约保证。"""
        flow_paths = self._flow_paths
        assert flow_paths is not None, (
            "step 链只能在 reset() 之后调用（k_paths 惰性初始化）"
        )
        return flow_paths

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """重置回合（legacy routingEnv.py:153-208）：Φ(s0)=0、速率扰动、k_paths 惰性。"""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._episode += 1
        self._last_rd = np.zeros(self._n)
        self._last_rp = np.zeros(self._n)
        scale = 1.0
        for episode_cap, value in self._change_flow_by_ep:
            if self._episode <= episode_cap:
                scale = value
        mean = self._rate_init * scale
        self._rates = np.clip(
            self._rng.normal(mean, self._rate_noise_scale),
            mean - self._rate_noise_clip,
            mean + self._rate_noise_clip,
        )
        if self._flow_paths is None:
            self._flow_paths = [
                self._paths_for(i, int(self._dst[i])) for i in range(self._n)
            ]
        info: dict[str, Any] = {
            "rd": np.zeros(self._n),
            "rp": np.zeros(self._n),
            "th": self._rates / self._th_max_rate,
            "reward_local": np.zeros(self._n),
            "reward_total": np.zeros(self._n),
            "r_mean": 0.0,
        }
        return self._build_state(), info

    def _extract_path_sequence(
        self, path: list[int], max_len: int = 8
    ) -> npt.NDArray[np.float64]:
        """提取路径节点序列特征: (max_len, 3)维，padding为0。

        每个节点3维特征: [rate/avgrate, last_rd, last_rp]
        不足max_len的用0填充（GRU会通过mask忽略padding）
        """
        seq = np.zeros((max_len, 3))
        for i, node in enumerate(path[:max_len]):
            seq[i, 0] = self._rates[node] / self._avgrate
            seq[i, 1] = self._last_rd[node]
            seq[i, 2] = self._last_rp[node]
        return seq.flatten()  # 展平为(max_len*3,)向量

    def _build_state(self) -> npt.NDArray[np.float64]:
        """(N, 78) 观测: 3维自身基础 + 72维路径序列(3路径×8节点×3特征) + 3维路径长度。

        GRU路径池化版本:
        - 自身基础(3维): rate/avgrate, last_rd, last_rp
        - 路径节点序列(72维): 3条路径 × 8节点 × 3特征（展平存储）
        - 路径长度(3维): 每条路径的实际节点数（GRU需要知道有效长度）

        路径序列保留完整的节点级信息，由策略网络的GRU模块池化。
        """
        flow_paths = self._paths()
        state = np.zeros((self._n, 78))
        max_path_len = 8

        for i in range(self._n):
            offset = 0

            # 自身基础特征(3维)
            state[i, 0] = self._rates[i] / self._avgrate
            state[i, 1] = self._last_rd[i]
            state[i, 2] = self._last_rp[i]
            offset = 3

            # 路径节点序列(72维: 3路径×8节点×3特征)
            for k, path in enumerate(flow_paths[i]):
                path_seq = self._extract_path_sequence(path, max_path_len)
                state[i, offset : offset + max_path_len * 3] = path_seq
                offset += max_path_len * 3
            offset = 75

            # 路径长度(3维)
            for k, path in enumerate(flow_paths[i]):
                state[i, offset + k] = min(len(path), max_path_len)

        return state

    @override
    def step(
        self, action: np.int64 | npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """一步（legacy routingEnv.py:452-464）：add_flows→reward→th 先算后存→change→state。

        action 为全节点动作向量 (N,)（D2 节点扇出语义）；形参名对齐 gym.Env.step。"""
        actions = np.asarray(action, dtype=np.int64)
        split = np.zeros((self._n, self._n_candidates))
        split[np.arange(self._n), actions] = 1.0
        self._add_flows(split)
        reward = self._get_reward(split)
        th = (
            self._rates / self._th_max_rate
        )  # N§4：先算后存（legacy 别名行为不同，记 issues）
        self._change_flows(reward.total)
        obs = self._build_state()
        truncated = self._t == self._episode_steps - 1  # 第 50 步 True
        self._t += 1
        info: dict[str, Any] = {
            "rd": reward.rd,
            "rp": reward.rp,
            "th": th,
            "reward_local": reward.local,
            "reward_total": reward.total,
            "r_mean": reward.r_mean,
        }
        return obs, reward.r_mean, False, truncated, info

    def _add_flows(self, split: npt.NDArray[np.float64]) -> None:
        """注入流量并传播（legacy routingEnv.py:210-266 的 dict 语义）。

        分流缝：split 为 (N, n_candidates) 任意非负比例矩阵（行和无需为 1），
        step 的 one-hot 是其特例——M2.5 比例向量直接走此入口。
        quirk：out_rate 首录胜出（setdefault，记 issues）。
        """
        flow_paths = self._paths()
        in_rate: list[dict[int, float]] = [{} for _ in range(self._n)]
        reminder: list[dict[int, int]] = [{} for _ in range(self._n)]
        out_rate: list[dict[int, float]] = [{} for _ in range(self._n)]
        for i in range(self._n):
            for k, path in enumerate(flow_paths[i]):
                key = i * self._n_candidates + k
                in_rate[path[0]][key] = float(self._rates[i] * split[i, k])
                for pos, node in enumerate(path):
                    reminder[node][key] = path[pos + 1] if pos < len(path) - 1 else -1
        for _ in range(self._propagation_rounds):
            for v in range(self._n):
                entries = in_rate[v]
                if not entries:
                    continue
                loss = mmk_loss(
                    sum(entries.values()), float(self._mu[v]), self._capacity
                )
                for key, ir in entries.items():
                    out = ir * (1.0 - loss)
                    out_rate[v].setdefault(key, out)
                    nxt = reminder[v][key]
                    if nxt != -1:
                        in_rate[nxt][key] = out
        self._in_rate = in_rate
        self._out_rate = out_rate

    def _get_reward(self, split: npt.NDArray[np.float64]) -> _StepReward:
        """奖励（legacy routingEnv.py:267-371）。

        quirks（均记 issues）：路径时延的 μ 取 flow 源节点（非路径节点 p）；
        out_rate 缺 key 默认 0.0（legacy 返回 None 潜在崩溃）；total 用 Python round(,2)。
        PBRS：local = r_base + gamma*G_cur − G_prev，G_prev = beta*last_rd+alpha*last_rp
        （reset 归零 → Φ(s0)=0，与 legacy 跨回合残留不同）。
        """
        flow_paths = self._paths()
        local = np.zeros(self._n)
        rd_new = np.zeros(self._n)
        rp_new = np.zeros(self._n)
        for i in range(self._n):
            rate = float(self._rates[i])
            mu_src = float(self._mu[i])
            path_delay = 0.0
            ava = 0.0
            for k, path in enumerate(flow_paths[i]):
                rate_k = rate * split[i, k]
                if rate_k == 0.0:
                    continue
                sp_delay = 0.0
                sp_ava = rate_k
                for p in path:
                    sp_delay += mmk_delay(
                        sum(self._in_rate[p].values()), mu_src, self._capacity
                    )
                    if p == path[-1]:
                        sp_ava = self._out_rate[p].get(i * self._n_candidates + k, 0.0)
                path_delay += (rate_k / rate) * sp_delay
                ava += sp_ava
            r_d = 1.0 - path_delay * mu_src / (self._n * self._capacity)
            r_p = ava / rate
            rd_new[i] = r_d
            rp_new[i] = r_p
            g_cur = self._beta_delay * r_d + self._alpha_plr * r_p
            g_prev = (
                self._beta_delay * self._last_rd[i] + self._alpha_plr * self._last_rp[i]
            )
            if self._pbrs:
                local[i] = g_cur + self._gamma * g_cur - g_prev
            else:
                local[i] = g_cur
        self._last_rd = rd_new
        self._last_rp = rp_new
        omega = self._rates / self._rates.sum()
        global_term = float((omega * local).sum())
        total = np.array(
            [
                round(self._mix_local * loc + self._mix_global * global_term, 2)
                for loc in local
            ]
        )
        return _StepReward(local, total, float(total.mean()), rd_new, rp_new)

    def _change_flows(self, total: npt.NDArray[np.float64]) -> None:
        """按 total 稳定升序调节半数流速率 ±change_flow_pct（legacy routingEnv.py:466-481）。"""
        for rank, node in enumerate(np.argsort(total, kind="stable")):
            factor = (
                1.0 - self._change_flow_pct
                if rank <= self._n / 2
                else 1.0 + self._change_flow_pct
            )
            self._rates[node] *= factor
