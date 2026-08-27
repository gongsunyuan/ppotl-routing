"""镜像 legacy 的 ActorCritic 网络构建器（M1-2，决策 D3/F2）。

legacy 出处：Fast_Adaptation.../01_数值实验/code/main.py:81-111——actor/critic 为独立
Sequential(Linear(287,128), PReLU, Linear(128,64), PReLU, 头)，全部 Linear 用
kaiming_normal_(mode="fan_out", nonlinearity="relu")（仅 weight，legacy 同口径）；
PReLU 保持默认初始化（单参数 0.25，不耗 rng）。

seed 控制：kaiming/uniform 经 torch.Generator 消耗，同 seed 两次 build 逐位相等；
bias 复刻 nn.Linear.reset_parameters 的 U(-1/sqrt(fan_in), 1/sqrt(fan_in))（PyTorch
同式，但换成 generator 控制，否则 bias 漂自全局 rng、复现性破坏）。
隐层默认读 config/default.yaml net.layers（论文超参不在代码硬编码）。
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Generator, nn

from trl_sb3.common.config import load_config


def _hidden_layers(hidden: Sequence[int] | None) -> tuple[int, ...]:
    """隐层元：显式传入优先，否则读 config net.layers 去掉首元素（首元素是输入维 287）。"""
    if hidden is not None:
        return tuple(hidden)
    return tuple(load_config()["net"]["layers"][1:])


def _init_linear(module: nn.Linear, generator: Generator | None) -> None:
    """legacy 口径初始化：kaiming(fan_out, relu) weight + Linear 默认分布 bias（generator 控制）。"""
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu", generator=generator)
    bound = 1.0 / module.weight.size(1) ** 0.5
    nn.init.uniform_(module.bias, -bound, bound, generator=generator)


def _make_net(dims: Sequence[int], generator: Generator | None) -> nn.Sequential:
    """按 dims 链 Linear，层间 PReLU（头后无激活——actor 出 logits、critic 出 V 标量）。"""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.PReLU())
    net = nn.Sequential(*layers)
    for module in net:
        if isinstance(module, nn.Linear):
            _init_linear(module, generator)
    return net


def make_actor(
    obs_dim: int = 287,
    n_actions: int = 3,
    hidden: Sequence[int] | None = None,
    generator: Generator | None = None,
) -> nn.Sequential:
    """actor：躯干 287→128→64（PReLU）+ 头 Linear(64, n_actions)（输出 logits，
    softmax 由 SB3 CategoricalDistribution 的 logits 路径等价承担）。"""
    return _make_net([obs_dim, *_hidden_layers(hidden), n_actions], generator)


def make_critic(
    obs_dim: int = 287,
    hidden: Sequence[int] | None = None,
    generator: Generator | None = None,
) -> nn.Sequential:
    """critic：同躯干 + 头 Linear(64, 1)（V(s) 标量）。"""
    return _make_net([obs_dim, *_hidden_layers(hidden), 1], generator)


def build_nets(
    obs_dim: int,
    n_actions: int,
    seed: int | None = None,
    hidden: Sequence[int] | None = None,
) -> tuple[nn.Sequential, nn.Sequential]:
    """统一入口：返回 (actor, critic)，共享一个 Generator（先 actor 后 critic 定序消耗，
    同 seed 逐位复现；seed=None 走全局 rng）。"""
    generator = None if seed is None else Generator().manual_seed(seed)
    return (
        make_actor(obs_dim, n_actions, hidden=hidden, generator=generator),
        make_critic(obs_dim, hidden=hidden, generator=generator),
    )
