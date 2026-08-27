"""SB3 2.7.0 ActorCriticPolicy 子类（M1-2，决策 D3）：pi/vf 完全独立两网接入，
绕过 MlpExtractor 默认 Tanh/orthogonal 路径。

接线点（venv stable-baselines3 2.7.0 源码核对，common/policies.py 行号）：
- ``__init__`` :448-535 末行 ``self._build(lr_schedule)`` → 动态分派到本类覆写；
  本类强制 share_features_extractor=False（:463 默认 True）与 ortho_init=False
  （:455 默认 True）——extract_features :660-682 走 pi/vf 双抽取，正交初始化整块不触发。
- ``_build_mlp_extractor`` :570-583 原建 MlpExtractor；本类换 SeparatePiVfNets
  （nets.py 两网：actor 287→128→64→3 logits、critic →1，kaiming fan_out）。
- ``_build`` :585-634 原实现建 action_net/value_net（:605/:609）+ ortho_init（:610-631）
  + optimizer（:634）；本类覆写为 action_net/value_net=Identity（头已在两网内，
  logits 直达 CategoricalDistribution.proba_distribution(action_logits=...)，
  与 legacy main.py:111 Softmax+Categorical(probs) 数值等价）并直接建 optimizer。
- 消费面零覆写：forward :636-658 / evaluate_actions :719-741 / get_distribution :743-752 /
  predict_values :754-763 均经 self.mlp_extractor.forward_actor / forward_critic
  （:650-651、:735-736、:751、:762）——自定义 extractor 只需这四个入口 + latent_dim_*。

legacy 结构对照 main.py:81-111：actor/critic 同构独立 Sequential、PReLU、
kaiming_normal_(fan_out, relu)；差异仅 SB3 侧 optimizer/保存格式。

冻结锚点（M2-1/D4）：config freeze.frozen_layers 的 actor.0/.1/.2 对应
state_dict 前缀 ``mlp_extractor.actor.*`` / ``mlp_extractor.critic.*``。
"""

from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy as SB3ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor, nn

from trl_sb3.common.config import load_config
from trl_sb3.policy.nets import build_nets


class SeparatePiVfNets(nn.Module):
    """MlpExtractor 替身：pi/vf 独立两网（无共享参数），暴露 SB3 消费的
    forward_actor/forward_critic 与 latent_dim_pi/latent_dim_vf（头输出维）。"""

    def __init__(self, actor: nn.Sequential, critic: nn.Sequential) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        # Sequential.__getitem__ 静态返回 Sequential | Module（torch 2.13 stub），
        # out_features 只在头上——isinstance 收窄到 Linear 后取值。
        actor_head = actor[-1]
        critic_head = critic[-1]
        assert isinstance(actor_head, nn.Linear) and isinstance(critic_head, nn.Linear)
        self.latent_dim_pi = actor_head.out_features
        self.latent_dim_vf = critic_head.out_features

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: Tensor) -> Tensor:
        return self.actor(features)

    def forward_critic(self, features: Tensor) -> Tensor:
        return self.critic(features)


def _hidden_from_config(features_dim: int) -> tuple[int, ...]:
    """隐层读 config net.layers（论文超参唯一出处），并校验输入维与观测空间一致。"""
    net_cfg: dict[str, Any] = load_config()["net"]
    layers: list[int] = list(net_cfg["layers"])
    if layers[0] != features_dim:
        raise ValueError(f"config net.layers 输入维 {layers[0]} != 观测特征维 {features_dim}")
    return tuple(layers[1:])


class ActorCriticPolicy(SB3ActorCriticPolicy):
    """TRL 路由策略：legacy 同构 pi/vf 独立两网，PPO(policy=ActorCriticPolicy) 直接可用。"""

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        action_space: spaces.Space[Any],
        lr_schedule: Schedule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("share_features_extractor", False)
        kwargs.setdefault("ortho_init", False)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """覆写 policies.py:570-583：nets.py 两网（kaiming 全局 rng，PPO seed 可复现）。
        头维 = 类别数（get_action_dim 对 Discrete 返回动作向量维 1，此处不适用）。"""
        assert isinstance(self.action_space, spaces.Discrete), "TRL 策略只支持 Discrete(K)"
        hidden = _hidden_from_config(self.features_dim)
        actor, critic = build_nets(self.features_dim, int(self.action_space.n), hidden=hidden)
        self.mlp_extractor = SeparatePiVfNets(actor, critic).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        """覆写 policies.py:585-634：头已在两网内（action_net/value_net=Identity，无参数），
        跳过 ortho_init 块 :610-631；optimizer 直建 Adam（D3 单 Adam——freeze.rebuild_optimizer
        同口径，optimizer_class 动态派发从不用）。"""
        self._build_mlp_extractor()
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
