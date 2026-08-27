"""NodeFanVecEnv：node-fanned VecEnv（M1-1，决策 D2——本次重构的核心创新点）。

把**一个** RoutingEnv（obs 已是 (N,287) 节点行堆叠）包装成 SB3 VecEnv：底层一步推进
扇出 N slot，N=底层拓扑节点数。SB3 契约出处（venv 内 stable-baselines3 2.7.0 源码核对）：
- step→step_async/step_wait 分派：common/vec_env/base_vec_env.py:214-222
- done 换算 / terminal_observation / 自动 reset：common/vec_env/dummy_vec_env.py:59-71
  （done = terminated or truncated；每步写 TimeLimit.truncated = truncated and not terminated；
   done 时 info 存末观测再 reset，返回 reset 后的新回合首观测）
- 终态 GAE bootstrap 消费点：common/on_policy_algorithm.py:236-245——当且仅当 slot info 同时
  含 terminal_observation 与 TimeLimit.truncated=True 时 rewards[idx] += gamma*V(terminal_obs)，
  本类给每 slot 放**自己那行**末观测（(287,)）

D2 语义等价性论证要点：legacy 逐节点训练样本 = 每 step 的 N 个节点观测切片 + 广播奖励
addBufferRD(rewards/node_num)（F13）。本 VecEnv 的 rollout=50×N 样本中，每 slot 的
log_prob/ratio/clip 与 legacy 逐节点列切片同构 partition（同一标量 r_mean 广播 → 优势
同构；锁步 done/reset → 回合边界一致），故 PPO 更新在分布上与 legacy 等价。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from trl_sb3.env.routing_env import RoutingEnv

# 任务 D2 规定的公共别名 → RoutingEnv 私有属性名（底层 env 不可改，M0 既成事实）。
_ATTR_ALIASES: dict[str, str] = {"n": "_n", "mu": "_mu", "pbrs": "_pbrs"}


class NodeFanVecEnv(VecEnv):
    """单底层 RoutingEnv 的锁步扇出 VecEnv：num_envs=N，obs 行 i 即 slot i 的观测。"""

    _actions: npt.NDArray[np.int64]

    def __init__(self, env: RoutingEnv) -> None:
        # 先绑底层再 super()：基类 __init__ 会调 get_attr("render_mode")（base_vec_env.py:75-79）
        self._env = env
        super().__init__(env._n, env.observation_space, env.action_space)

    def step_async(self, actions: npt.NDArray[np.int64]) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """底层一步：r_mean 广播、锁步 done、truncated 时自动 reset（dummy_vec_env.py:59-71 同构）。

        终态语义（D1 修 F5）：reward 用本步的，obs 返回 reset 后新回合首观测，每 slot info
        带 terminal_observation=本步末观测行 + TimeLimit.truncated。
        """
        obs, reward, terminated, truncated, info = self._env.step(self._actions)
        done = bool(terminated or truncated)
        truncated_only = bool(truncated and not terminated)
        final_obs: npt.NDArray[np.float64] | None = None
        if done:
            final_obs = obs
            obs, reset_info = self._env.reset()
            self.reset_infos = [dict(reset_info) for _ in range(self.num_envs)]
        rewards = np.full(self.num_envs, reward, dtype=np.float64)
        dones = np.full(self.num_envs, done, dtype=bool)
        infos: list[dict[str, Any]] = []
        for slot in range(self.num_envs):
            slot_info = dict(info)  # 每 slot 独立 dict：VecMonitor 会逐 slot 改写
            slot_info["TimeLimit.truncated"] = truncated_only
            if final_obs is not None:
                slot_info["terminal_observation"] = final_obs[slot]
            infos.append(slot_info)
        return obs, rewards, dones, infos

    def reset(self, *, seed: int | None = None) -> VecEnvObs:
        """原子 reset：底层 reset 一次，(N,287) 行即 N slot 观测（错做 N 次则 rng 流推进 N 次，
        与裸 env 单次 reset 的逐位相等性即破——测试以此证原子性）。seed 参数优先，否则消费
        seed() 存入 _seeds[0] 的种子（单底层 env 只有 _seeds[0] 有意义），用后即清。"""
        effective_seed = seed if seed is not None else self._seeds[0]
        obs, info = self._env.reset(seed=effective_seed)
        self.reset_infos = [dict(info) for _ in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return obs

    def close(self) -> None:
        self._env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """底层属性 → N 份相同值；n/mu/pbrs 走 D2 公共别名。"""
        slots = list(self._get_indices(indices))
        value = self._env.get_wrapper_attr(_ATTR_ALIASES.get(attr_name, attr_name))
        return [value for _ in slots]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        # 单底层 env：indices 无独立落点，无论指向哪些 slot 都只赋一次
        setattr(self._env, attr_name, value)

    def env_method(
        self, method_name: str, *method_args: Any, indices: VecEnvIndices = None, **method_kwargs: Any
    ) -> list[Any]:
        slots = list(self._get_indices(indices))
        result = self._env.get_wrapper_attr(method_name)(*method_args, **method_kwargs)
        return [result for _ in slots]

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper[Any, Any, Any, Any]], indices: VecEnvIndices = None
    ) -> list[bool]:
        # 底层裸 RoutingEnv 无 gymnasium Wrapper，恒 False
        return [False for _ in self._get_indices(indices)]
