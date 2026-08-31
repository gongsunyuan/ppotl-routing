"""SB3 2.7.0 ActorCriticPolicy 子类（M1-2，决策 D3）：pi/vf 完全独立两网接入，
绕过 MlpExtractor 默认 Tanh/orthogonal 路径。

接线点（venv stable-baselines3 2.7.0 源码核对，common/policies.py 行号）：
- ``__init__`` :448-535 末行 ``self._build(lr_schedule)`` → 动态分派到本类覆写；
  本类强制 share_features_extractor=False（:463 默认 True）与 ortho_init=False
  （:455 默认 True）——extract_features :660-682 走 pi/vf 双抽取，正交初始化整块不触发。
- ``_build_mlp_extractor`` :570-583 原建 MlpExtractor；本类换 SeparatePiVfNets
  （nets.py 两网：actor 287→128→64→3 logits、critic →1，kaiming fan_out）。
  隐层来源两级：policy_kwargs ``net_layers``（per-run 覆盖缝——runner 从 spec.config
  注入，SB3 把 policy_kwargs pickle 进 ckpt、PPO.load 后 ``_setup_model`` 以
  ``**policy_kwargs`` 重建，覆盖随 ckpt 往返）优先；缺省回落全局 config
  net.layers（论文超参唯一出处）。
- ``_build`` :585-634 原实现建 action_net/value_net（:605/:609）+ ortho_init（:610-631）
  + optimizer（:634）；本类覆写为 action_net/value_net=Identity（头已在两网内，
  logits 直达 CategoricalDistribution.proba_distribution(action_logits=...)，
  与 legacy main.py:111 Softmax+Categorical(probs) 数值等价）并直接建 optimizer
  （缺省 D3 单 Adam；lr_critic_ratio 双参数组可选，缝 3——DEVIATIONS #3 legacy
  方向已钉死：actor 6e-6 / critic 6e-7 = 10×，即 ratio=critic/actor=0.1）。
- 消费面零覆写：forward :636-658 / evaluate_actions :719-741 / get_distribution :743-752 /
  predict_values :754-763 均经 self.mlp_extractor.forward_actor / forward_critic
  （:650-651、:735-736、:751、:762）——自定义 extractor 只需这四个入口 + latent_dim_*。

legacy 结构对照 main.py:81-111：actor/critic 同构独立 Sequential、PReLU、
kaiming_normal_(fan_out, relu)；差异仅 SB3 侧 optimizer/保存格式。

冻结锚点（M2-1/D4）：config freeze.frozen_layers 的 actor.0/.1/.2 对应
state_dict 前缀 ``mlp_extractor.actor.*`` / ``mlp_extractor.critic.*``。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, override

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
        self.actor: nn.Sequential = actor
        self.critic: nn.Sequential = critic
        # Sequential.__getitem__ 静态返回 Sequential | Module（torch 2.13 stub），
        # out_features 只在头上——isinstance 收窄到 Linear 后取值。
        actor_head = actor[-1]
        critic_head = critic[-1]
        assert isinstance(actor_head, nn.Linear) and isinstance(critic_head, nn.Linear)
        self.latent_dim_pi: int = actor_head.out_features
        self.latent_dim_vf: int = critic_head.out_features

    @override
    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: Tensor) -> Tensor:
        return self.actor(features)

    def forward_critic(self, features: Tensor) -> Tensor:
        return self.critic(features)


def _hidden_from_config(features_dim: int, layers: Sequence[int] | None = None) -> tuple[int, ...]:
    """隐层来源二选一：显式 layers（policy_kwargs net_layers，per-run 覆盖）或
    config net.layers（论文超参唯一出处）；输入维与观测空间一致两条路径同校验。"""
    net_layers = tuple(load_config()["net"]["layers"]) if layers is None else tuple(layers)
    if net_layers[0] != features_dim:
        raise ValueError(f"隐层输入维 {net_layers[0]} != 观测特征维 {features_dim}")
    return net_layers[1:]


def split_actor_critic(named_params: Iterable[tuple[str, Tensor]]) -> tuple[list[Tensor], list[Tensor]]:
    """双 lr 分区（缝 3）：按参数名前缀二分——actor=mlp_extractor.actor.*、
    critic=mlp_extractor.critic.*（当前策略参数恰好全落这两前缀：action_net/
    value_net=Identity、FlattenExtractor 均无参数）。前缀之外出现任何参数
    （如未来可学习 features_extractor）直接 ValueError——宁可响亮失败，
    不可静默漏出优化。freeze.rebuild_optimizer 以可训练子集同口径复用。"""
    actor_params: list[Tensor] = []
    critic_params: list[Tensor] = []
    strays: list[str] = []
    for name, param in named_params:
        if name.startswith("mlp_extractor.actor."):
            actor_params.append(param)
        elif name.startswith("mlp_extractor.critic."):
            critic_params.append(param)
        else:
            strays.append(name)
    if strays:
        raise ValueError(f"双 lr 分区不完备：actor/critic 前缀之外的参数 {strays}（须显式归组，勿静默漏优化）")
    return actor_params, critic_params


class DualLrAdam(th.optim.Adam):
    """双参数组 Adam（缝 3）：组 0=actor（基 lr）、组 1=critic（基 lr × ratio）。

    SB3 2.7.0 坑防御（venv 源码核对）：PPO.train() 每次先 _update_learning_rate
    （ppo.py:191 → common/utils.py update_learning_rate）把**所有** param_groups
    的 lr 重置为同一 schedule 基值——纯 Adam 双组的 critic lr 会被静默抬到 actor
    基值。本类在每次 step() 前按组 0 当前 lr 重申组 1 比例：critic 恒以
    actor_lr × ratio 步进；基值来自 schedule 时两组同取 schedule 当前值
    （× 比例），是单 Adam「lr 取 schedule 首值」口径向双组的自然推广。
    SB3 ckpt 只存 optimizer state_dict（组数/组序由 _build 从 policy_kwargs
    重建，PyTorch load_state_dict 要求两侧组数一致），本类随 _build 重建。

    构造参数 4 个（>3）的理由：镜像 th.optim.Adam 自身 (params, lr) 签名再加
    双网拆分与比例，四者同属一个优化器规格，拆 dataclass 反而隔一层。
    """

    def __init__(
        self,
        actor_params: Sequence[Tensor],
        critic_params: Sequence[Tensor],
        lr: float,
        lr_critic_ratio: float,
        **optimizer_kwargs: Any,
    ) -> None:
        super().__init__(
            [
                {"params": list(actor_params), "lr": lr},
                {"params": list(critic_params), "lr": lr * lr_critic_ratio},
            ],
            lr=lr,
            **optimizer_kwargs,
        )
        self._lr_critic_ratio: float = float(lr_critic_ratio)

    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self.param_groups[1]["lr"] = self.param_groups[0]["lr"] * self._lr_critic_ratio
        return super().step(closure)


class ActorCriticPolicy(SB3ActorCriticPolicy):
    """TRL 路由策略：legacy 同构 pi/vf 独立两网，PPO(policy=ActorCriticPolicy) 直接可用。

    自有 policy_kwargs 两键（SB3 基类签名定长，须在 super().__init__ 前 pop）：
    ``net_layers`` 全链生效（_build_mlp_extractor 构网）；``lr_critic_ratio``
    双 lr 开关（缝 3）：None=D3 单 Adam 缺省（逐位同旧行为）；float r=_build
    建双参数组 Adam（critic 组 lr = actor 组 × r；0.1 复刻 legacy
    actor/critic=10×，DEVIATIONS #3）。二者随 SB3 ckpt 往返。"""

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        action_space: spaces.Space[Any],
        lr_schedule: Schedule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        net_layers = kwargs.pop("net_layers", None)
        lr_critic_ratio = kwargs.pop("lr_critic_ratio", None)
        # 属性须先于 super().__init__ 置位：其末行 _build(lr_schedule) 即消费 net_layers。
        self.net_layers: tuple[int, ...] | None = None if net_layers is None else tuple(net_layers)
        self.lr_critic_ratio: float | None = None if lr_critic_ratio is None else float(lr_critic_ratio)
        kwargs.setdefault("share_features_extractor", False)
        kwargs.setdefault("ortho_init", False)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    @override
    def _build_mlp_extractor(self) -> None:
        """覆写 policies.py:570-583：nets.py 两网（kaiming 全局 rng，PPO seed 可复现）。
        头维 = 类别数（get_action_dim 对 Discrete 返回动作向量维 1，此处不适用）；
        隐层读 self.net_layers（缺省回落全局 config，见 _hidden_from_config）。"""
        assert isinstance(self.action_space, spaces.Discrete), "TRL 策略只支持 Discrete(K)"
        hidden = _hidden_from_config(self.features_dim, self.net_layers)
        actor, critic = build_nets(self.features_dim, int(self.action_space.n), hidden=hidden)
        # 结构化替身（非 MlpExtractor 子类，D4 决策）：SB3 按 forward_actor/critic +
        # latent_dim_pi/vf 鸭子类型消费——显式窄注解供 freeze 等消费方取 .actor/.critic。
        self.mlp_extractor: SeparatePiVfNets = SeparatePiVfNets(actor, critic).to(self.device)

    @override
    def _build(self, lr_schedule: Schedule) -> None:
        """覆写 policies.py:585-634：头已在两网内（action_net/value_net=Identity，无参数），
        跳过 ortho_init 块 :610-631。optimizer：D3 单 Adam 缺省 + lr_critic_ratio
        双组可选（缝 3）——actor 组 lr=lr_schedule(1)、critic 组 = actor × ratio
        （0.1 复刻 legacy actor/critic=10×：actor 6e-6 / critic 6e-7，DEVIATIONS #3；
        schedule 型 lr 两组同取首值，单 Adam 同口径）。分区按名前缀
        mlp_extractor.actor.*/critic.*（split_actor_critic，前缀之外响亮报错）；
        freeze.rebuild_optimizer 同口径镜像（只取可训练子集）。"""
        self._build_mlp_extractor()
        self.action_net: nn.Module = nn.Identity()
        self.value_net: nn.Module = nn.Identity()
        if self.lr_critic_ratio is None:
            self.optimizer: th.optim.Adam = th.optim.Adam(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
        else:
            actor_params, critic_params = split_actor_critic(self.named_parameters())
            self.optimizer = DualLrAdam(
                actor_params, critic_params, lr_schedule(1), self.lr_critic_ratio, **self.optimizer_kwargs
            )
