"""臂训练 runner（M2-3）：消融臂 → 训练 env → PPO（新建/加载）→ learn+日志 → 产物契约。

臂因子表 ARM_FACTORS 逐条对齐计划 §4（pretrain/freeze/pbrs）；run_id 全经
make_run_id（确定性，断点续跑/DONE 跳过依赖它）；产物：metrics.csv（每步一行）、
eval.json（周期曲线+终评聚合）、manifest.json、ckpt、DONE 最后 / FAILED 带 traceback。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

from trl_sb3.common.config import load_config, resolve_path
from trl_sb3.common.envs import build_routing_env
from trl_sb3.common.logging_utils import MetricsCSVWriter, make_run_id, run_dir, write_manifest
from trl_sb3.common.run_artifacts import (
    METRICS_COLUMNS,
    build_manifest,
    is_done,
    mark_done,
    mark_failed,
)
from trl_sb3.env.node_fan_vec import NodeFanVecEnv
from trl_sb3.eval.evaluate import EVAL_AGG_KEYS, greedy_eval, ppo_policy
from trl_sb3.policy.freeze import apply_freeze, rebuild_optimizer
from trl_sb3.policy.policy import ActorCriticPolicy

# 计划 §4 因子表（pretrain/freeze/pbrs）。A1b 等总预算对照与 A1 同因子
# （预算差由 sweep #9 的 episodes 承担，不入因子）；A3b = 随机网络冻结对照。
ARM_FACTORS: dict[str, dict[str, bool]] = {
    "A1": {"pretrain": False, "freeze": False, "pbrs": False},
    "A1b": {"pretrain": False, "freeze": False, "pbrs": False},
    "A2": {"pretrain": True, "freeze": False, "pbrs": False},
    "A3": {"pretrain": True, "freeze": True, "pbrs": False},
    "A3b": {"pretrain": False, "freeze": True, "pbrs": False},
    "A4": {"pretrain": False, "freeze": False, "pbrs": True},
    "A5": {"pretrain": True, "freeze": False, "pbrs": True},
    "A6": {"pretrain": True, "freeze": True, "pbrs": True},
}


@dataclass(frozen=True, slots=True)
class _RunSpec:
    """一次训练 run 的全部决定因素（frozen：构造后不可变，进 manifest 的口径）。"""

    arm: str
    topo: str
    rate: int
    seed: int
    pbrs: bool
    freeze: bool
    source_run_id: str | None
    episodes: int
    out_root: Path | None
    ckpts_dir: Path | None
    device: str
    config: dict[str, Any]


class _RunLogger(BaseCallback):
    """训练日志：on_step 取 infos[0] 的 rd/rp/th（per-node 均值）与 r_mean 写 csv 行；
    回合边界 = info["TimeLimit.truncated"]（NodeFanVecEnv 每步恒置该键）；每
    eval_interval 回合在干净评估 env 上 greedy_eval 记曲线点（评估/训练 env 严格分离）。"""

    def __init__(
        self,
        writer: MetricsCSVWriter,
        *,
        arm: str,
        topo: str,
        rate: int,
        seed: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__(verbose=0)
        self._writer = writer
        self._arm = arm
        self._topo = topo
        self._rate = rate
        self._seed = seed
        self._config = config
        self._interval = int(config["eval"]["eval_interval"])
        self._episode = 0
        self._step = 0
        self.curve: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._writer.write_row(
            {
                "arm": self._arm,
                "topo": self._topo,
                "rate": self._rate,
                "seed": self._seed,
                "episode": self._episode,
                "step": self._step,
                "rd": float(info["rd"].mean()),
                "rp": float(info["rp"].mean()),
                "th": float(info["th"].mean()),
                "r_mean": float(info["r_mean"]),
            }
        )
        if info["TimeLimit.truncated"]:
            self._episode += 1
            self._step = 0
            if self._episode % self._interval == 0:
                result = greedy_eval(ppo_policy(self.model), self._topo, self._rate, config=self._config)
                point: dict[str, float] = {"episode": self._episode}
                point.update({key: result[key] for key in EVAL_AGG_KEYS})
                self.curve.append(point)
        else:
            self._step += 1
        return True


def _build_model(vec: VecMonitor, spec: _RunSpec, ckpts_dir: Path) -> PPO:
    """预训练臂 PPO.load 源 ckpt（加载后重算整批 batch_size；冻结臂施加 D4 冻结）；
    其余臂从 config ppo 节全参新建（batch_size=null → n_steps×N 整批，D2）。"""
    if spec.source_run_id is not None:
        model = PPO.load(ckpts_dir / f"{spec.source_run_id}.zip", env=vec, device=spec.device)
        # D2 整批：batch_size 必须等于 n_steps×**目标**拓扑节点数。ckpt 恢复的是源拓扑
        # 的整批（如 CERNET 41 节点的 2050），对 N=11 的 rollout（550）会在 PPO.train
        # 的 mini-batch 切分上出错——加载后按目标 N 重算。
        model.batch_size = model.n_steps * vec.num_envs
        if spec.freeze:  # A3/A6：加载后冻结
            apply_freeze(model.policy, seed=spec.seed)
            rebuild_optimizer(model)
        return model
    ppo_cfg = spec.config["ppo"]
    n_nodes = vec.num_envs
    batch_size = ppo_cfg["n_steps"] * n_nodes if ppo_cfg["batch_size"] is None else int(ppo_cfg["batch_size"])
    model = PPO(
        policy=ActorCriticPolicy,
        env=vec,
        n_steps=ppo_cfg["n_steps"],
        n_epochs=ppo_cfg["n_epochs"],
        batch_size=batch_size,
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        learning_rate=ppo_cfg["learning_rate"],
        normalize_advantage=ppo_cfg["normalize_advantage"],
        verbose=0,
        seed=spec.seed,
        device=spec.device,
    )
    if spec.freeze:  # A3b：随机网络冻结
        apply_freeze(model.policy, seed=spec.seed)
        rebuild_optimizer(model)
    return model


def run_arm(
    arm: str,
    *,
    topo: str,
    rate: int,
    seed: int,
    source_run_id: str | None = None,
    episodes: int | None = None,
    out_root: str | Path | None = None,
    device: str = "cpu",
    config: dict[str, Any] | None = None,
) -> Path:
    """跑一个消融臂并落产物（返回 run 目录）。未知臂 ValueError；episodes 缺省
    config adapt.episodes；out_root/ckpts_dir 缺省 config paths 节。"""
    cfg = load_config() if config is None else config
    if arm not in ARM_FACTORS:
        raise ValueError(f"未知消融臂 {arm!r}；可用：{sorted(ARM_FACTORS)}")
    factors = ARM_FACTORS[arm]
    return _train_and_write(
        _RunSpec(
            arm=arm,
            topo=topo,
            rate=int(rate),
            seed=seed,
            pbrs=factors["pbrs"],
            freeze=factors["freeze"],
            source_run_id=source_run_id if factors["pretrain"] else None,
            episodes=int(cfg["adapt"]["episodes"]) if episodes is None else int(episodes),
            out_root=Path(out_root) if out_root is not None else None,
            ckpts_dir=None,
            device=device,
            config=cfg,
        )
    )


def _train_and_write(spec: _RunSpec) -> Path:
    """训练引擎：is_done 幂等跳过 → 全流程（训练+日志 → 终评 eval.json → manifest →
    ckpt → DONE 最后）→ 异常 mark_failed 后 re-raise。"""
    run_id = make_run_id(
        arm=spec.arm,
        topo=spec.topo,
        rate=spec.rate,
        seed=spec.seed,
        pbrs=spec.pbrs,
        freeze=spec.freeze,
        pretrain=spec.source_run_id or None,
    )
    out_root = spec.out_root if spec.out_root is not None else resolve_path(spec.config["paths"]["runs_dir"])
    ckpts_dir = spec.ckpts_dir if spec.ckpts_dir is not None else resolve_path(spec.config["paths"]["ckpts_dir"])
    directory = run_dir(out_root, run_id)
    if is_done(directory):
        return directory
    try:
        env = build_routing_env(spec.topo, spec.rate, pbrs=spec.pbrs, seed=spec.seed, config=spec.config)
        vec = VecMonitor(NodeFanVecEnv(env))
        model = _build_model(vec, spec, ckpts_dir)
        steps_per_episode = int(spec.config["env"]["episode_steps"])
        # truncate-on-restart（issues.md M3）：无 DONE 的中断 run 复跑时，旧 metrics.csv
        # 会被 append 模式重复累积 episode 行——先删，保证每 run 的 metrics 行集与本次执行一致。
        (directory / "metrics.csv").unlink(missing_ok=True)
        with MetricsCSVWriter(directory / "metrics.csv", METRICS_COLUMNS) as writer:
            logger = _RunLogger(
                writer, arm=spec.arm, topo=spec.topo, rate=spec.rate, seed=spec.seed, config=spec.config
            )
            model.learn(
                total_timesteps=spec.episodes * steps_per_episode * vec.num_envs, callback=logger
            )
        final_eval = greedy_eval(ppo_policy(model), spec.topo, spec.rate, config=spec.config)
        write_manifest(
            directory / "eval.json",
            {"curve": logger.curve, "final": {key: final_eval[key] for key in EVAL_AGG_KEYS}},
        )
        manifest = build_manifest(
            run_id,
            spec.arm,
            spec.topo,
            spec.rate,
            spec.seed,
            factors={"pretrain": spec.source_run_id is not None, "freeze": spec.freeze, "pbrs": spec.pbrs},
            source_run_id=spec.source_run_id,
            episodes=spec.episodes,
            device=spec.device,
            config=spec.config,
        )
        write_manifest(directory / "manifest.json", manifest)
        ckpts_dir.mkdir(parents=True, exist_ok=True)
        model.save(ckpts_dir / run_id)  # 全部训练臂都存 ckpt（<run_id>.zip）
        mark_done(directory)
    except Exception as exc:
        mark_failed(directory, exc)
        raise
    return directory
