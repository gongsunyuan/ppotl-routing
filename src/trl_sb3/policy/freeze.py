"""arm3/6 冻结语义（M2-1，决策 D4）：actor 躯干冻结、输出头 kaiming 重初始化、
critic 全训练，优化器在冻结后重建。

D4 语义：冻结 = ``requires_grad_(False)``，优化器冻结后构建。SB3 2.7.0 PPO.train
走 ``loss.backward()`` + ``optimizer.step()``（ppo/ppo.py:274-278），
``requires_grad=False`` 参数不进计算图、``.grad`` 恒 None → Adam 零更新零漂移，
无需 grad hook；重建后的优化器只含可训练参数，冻结参数连 optimizer 状态都不占。

F6/F7 考古依据：legacy main.py:290-295 死注释（索引推算为"冻结 actor 前层"）从未
真正运行部分冻结（F7）；本模块按其意图新定义——躯干冻结、头重初始化再训练、
critic 全训练（F7），语义以 config/default.yaml freeze 节为唯一出处。

冻结锚点映射（config freeze.frozen_layers 条目 ``<net>.N`` → policy 模块）：
    actor.N  → policy.mlp_extractor.actor[N]   （nn.Sequential 索引）
    critic.N → policy.mlp_extractor.critic[N]
当前 config [actor.0, actor.1, actor.2, actor.3] = Linear(287,128) / PReLU /
Linear(128,64) / PReLU（D4：除输出头外躯干全冻结）；头 = actor[-1] = actor[4]
Linear(64,3) 重初始化后可训练。
"""

from __future__ import annotations

from typing import Optional

import torch as th
from stable_baselines3 import PPO
from torch import Generator
from torch.nn import Linear

from trl_sb3.common.config import load_config
from trl_sb3.policy.nets import _init_linear
from trl_sb3.policy.policy import ActorCriticPolicy


def _module_at(policy: ActorCriticPolicy, anchor: str) -> th.nn.Module:
    """锚点解析："actor.N"→mlp_extractor.actor[N]；非法网名/索引直接 KeyError/IndexError。"""
    net_name, _, idx = anchor.partition(".")
    nets = {"actor": policy.mlp_extractor.actor, "critic": policy.mlp_extractor.critic}
    return nets[net_name][int(idx)]


def apply_freeze(policy: ActorCriticPolicy, seed: Optional[int] = None) -> None:
    """按 config freeze 节施加 D4 冻结（幂等：重复调用 requires_grad 状态不变）。

    - frozen_layers 逐锚点 requires_grad_(False)（躯干冻结）；
    - reinit_head：头 Linear(64,3) 先 legacy 口径 kaiming(fan_out, relu) 重初始化
      （nets._init_linear，weight+bias）再保持可训练；
    - critic_trainable：critic 全部可训练。
    seed 经 torch.Generator 控制重初始化，Generator 随头参数设备创建（CPU/CUDA，
    CPU Generator 对 CUDA 张量会 RuntimeError）；跨设备种子复现不保证逐位一致，
    同设备同 seed 一致（None 走全局 rng）。
    """
    freeze_cfg = load_config()["freeze"]
    for anchor in freeze_cfg["frozen_layers"]:
        _module_at(policy, anchor).requires_grad_(False)
    if freeze_cfg["reinit_head"]:
        head = policy.mlp_extractor.actor[-1]
        assert isinstance(head, Linear), f"actor[-1] 应为输出头 Linear，got {type(head).__name__}"
        generator = None if seed is None else Generator(device=head.weight.device.type).manual_seed(seed)
        _init_linear(head, generator)
        head.requires_grad_(True)
    if freeze_cfg["critic_trainable"]:
        policy.mlp_extractor.critic.requires_grad_(True)


def rebuild_optimizer(model: PPO) -> None:
    """冻结后重建单 Adam（D3 单 lr）：只含 requires_grad 参数，lr 读
    model.learning_rate（float 直取；schedule 取首值，PPO 构造同口径）。"""
    lr = model.learning_rate if isinstance(model.learning_rate, (int, float)) else model.learning_rate(1)
    trainable = [p for p in model.policy.parameters() if p.requires_grad]
    model.policy.optimizer = th.optim.Adam(trainable, lr=lr)
