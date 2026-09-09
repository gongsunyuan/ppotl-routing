"""arm3/6 冻结语义（M2-1，决策 D4）：actor 躯干冻结、输出头 kaiming 重初始化、
critic 全训练，优化器在冻结后重建。

D4 语义：冻结 = ``requires_grad_(False)``，优化器冻结后构建。SB3 2.7.0 PPO.train
走 ``loss.backward()`` + ``optimizer.step()``（ppo/ppo.py:274-278），
``requires_grad=False`` 参数不进计算图、``.grad`` 恒 None → Adam 零更新零漂移，
无需 grad hook；重建后的优化器只含可训练参数，冻结参数连 optimizer 状态都不占。

F6/F7 考古依据：legacy main.py:290-295 死注释（索引推算为"冻结 actor 前层"）从未
真正运行部分冻结（F7）；本模块按其意图新定义——躯干冻结、头重初始化再训练、
critic 全训练（F7），语义默认以 config/default.yaml freeze 节为出处；
apply_freeze 另接受显式 freeze_cfg 注入（超参传播缝 2：runner 传
spec.config["freeze"]，per-run 覆盖优先于全局 config）。

冻结锚点映射（config freeze.frozen_layers 条目，点分段逐级整数下标）：
    actor.N    → policy.mlp_extractor.actor[N]        （nn.Sequential 索引）
    actor.1.N  → policy.mlp_extractor.actor[1][N]     （GRU 版层级，见下）
    critic.N   → policy.mlp_extractor.critic[N]
GRU 版（2026-09-02 起）actor = Sequential(FeatureExtractor, MLP)：actor.0 =
    PathGRU 特征提取器（可训练，不进冻结清单）、actor.1 = MLP Sequential。
当前 config [actor.1.0, actor.1.1, actor.1.2, actor.1.3] = Linear(99,128) /
    PReLU / Linear(128,64) / PReLU（MLP 躯干全冻结）；头 = actor[1][4]
    Linear(64,3) 重初始化后可训练。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch as th
from stable_baselines3 import PPO
from torch import Generator

from trl_sb3.common.config import load_config
from trl_sb3.policy.nets import _init_linear
from trl_sb3.policy.policy import ActorCriticPolicy, DualLrAdam, split_actor_critic


def _module_at(policy: ActorCriticPolicy, anchor: str) -> th.nn.Module:
    """锚点解析：点分段逐级整数下标——"actor.N"→mlp_extractor.actor[N]（扁平）、
    "actor.1.N"→mlp_extractor.actor[1][N]（GRU 版层级，actor.0=FeatureExtractor、
    actor.1=MLP Sequential）；非法网名/索引直接 KeyError/IndexError。"""
    parts = anchor.split(".")
    nets = {"actor": policy.mlp_extractor.actor, "critic": policy.mlp_extractor.critic}
    node: th.nn.Module = nets[parts[0]]
    for seg in parts[1:]:
        assert isinstance(node, th.nn.Sequential), (
            f"锚点 {anchor} 中途遇到非 Sequential：{type(node).__name__}"
        )
        node = node[int(seg)]
    return node


def _head_linear(actor: th.nn.Sequential) -> th.nn.Linear:
    """输出头解析：从 actor 容器沿末子模块下钻到最内层 Linear——GRU 版
    actor=Sequential(FeatureExtractor, MLP) 的头在 actor[1] 末位；扁平版
    （无嵌套）即 actor[-1] 自身。"""
    node: th.nn.Module = actor[-1]
    while isinstance(node, th.nn.Sequential):
        node = node[-1]
    assert isinstance(node, th.nn.Linear), (
        f"输出头应为 Linear，got {type(node).__name__}"
    )
    return node


def apply_freeze(
    policy: ActorCriticPolicy,
    seed: int | None = None,
    *,
    freeze_cfg: Mapping[str, Any] | None = None,
) -> None:
    """按 freeze 节施加 D4 冻结（幂等：重复调用 requires_grad 状态不变）。

    节来源两级（超参传播缝 2）：显式 freeze_cfg（runner 从 spec.config 注入，
    per-run 覆盖）优先；缺省回落全局 config freeze 节（既有调用方口径不变）。

    - frozen_layers 逐锚点 requires_grad_(False)（躯干冻结）；
    - reinit_head：头 Linear(64,3) 先 legacy 口径 kaiming(fan_out, relu) 重初始化
      （nets._init_linear，weight+bias）再保持可训练；
    - critic_trainable：critic 全部可训练。
    seed 经 torch.Generator 控制重初始化，Generator 随头参数设备创建（CPU/CUDA，
    CPU Generator 对 CUDA 张量会 RuntimeError）；跨设备种子复现不保证逐位一致，
    同设备同 seed 一致（None 走全局 rng）。
    """
    cfg = load_config()["freeze"] if freeze_cfg is None else freeze_cfg
    for anchor in cfg["frozen_layers"]:
        _module_at(policy, anchor).requires_grad_(False)
    if cfg["reinit_head"]:
        head = _head_linear(policy.mlp_extractor.actor)
        generator = (
            None
            if seed is None
            else Generator(device=head.weight.device.type).manual_seed(seed)
        )
        _init_linear(head, generator)
        head.requires_grad_(True)
    if cfg["critic_trainable"]:
        policy.mlp_extractor.critic.requires_grad_(True)


def rebuild_optimizer(model: PPO) -> None:
    """冻结后重建优化器（D3 单 Adam 缺省 + lr_critic_ratio 双组镜像，缝 3）：只含
    requires_grad 参数，lr 读 model.learning_rate（float 直取；schedule 取首值，
    PPO 构造同口径）作 actor/基 lr。ratio 读 model.policy.lr_critic_ratio（属性随
    SB3 ckpt 往返）：非 None 时镜像 policy._build 双组——可训练 actor 参数入 actor
    组（冻结躯干排除、输出头保留）、可训练 critic 全入 critic 组，critic 组
    lr = actor × ratio（0.1 = legacy actor/critic=10×，DEVIATIONS #3）；分区同
    split_actor_critic 口径，前缀之外的可训练参数响亮报错。"""
    lr = (
        model.learning_rate
        if isinstance(model.learning_rate, (int, float))
        else model.learning_rate(1)
    )
    policy = model.policy
    assert isinstance(policy, ActorCriticPolicy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    ratio = policy.lr_critic_ratio
    if ratio is None:
        policy.optimizer = th.optim.Adam(trainable, lr=lr)
        return
    actor_params, critic_params = split_actor_critic(
        (name, p) for name, p in policy.named_parameters() if p.requires_grad
    )
    policy.optimizer = DualLrAdam(actor_params, critic_params, lr, ratio)
