"""单 run 画像(Phase-0 §2.6):进程内预训练 + 计时探针,device=cpu vs cuda 对比。

不改动 src/:仅在**本进程**内包装 trl_sb3.train.runner 的模块级缝——
- runner.build_routing_env → _TimingEnv 代理(累计底层 RoutingEnv step/reset 墙钟);
- runner.greedy_eval      → 计时包装(调用次数+时长;覆盖训练内周期评估与终评);
- runner.PPO              → _TimingPPO 子类(learn / train 两相位累计)。
相位分解:total = setup + learn_total + final_eval;learn_total ⊇ ppo_train(优化)
+ periodic_eval + rollout(含 env_step 与前向);env_step/env_reset 由代理单列。

GPU 采样:nvidia-smi 子进程 1Hz 轮询全部 GPU(本机无 pynvml;CUDA 跑用
CUDA_VISIBLE_DEVICES 钉到空闲卡,分析时取该卡列)。

用法(工作目录 = 仓库根):
  .venv/bin/python profiling/profile_single.py --device cpu  --episodes 100 --rep 1
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python profiling/profile_single.py --device cuda --episodes 100 --rep 1
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python profiling/profile_single.py --device cuda --profiler-segment
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

import trl_sb3.train.runner as runner_mod
from trl_sb3.common.config import load_config
from trl_sb3.common.envs import build_routing_env
from trl_sb3.env.node_fan_vec import NodeFanVecEnv
from trl_sb3.train.pretrain import run_pretrain

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "profiling"
SEED = 777  # 画像专用种子,不与调参/试点流 {0..9,100..102} 相交


class _TimingEnv:
    """RoutingEnv 计时代理:只拦 step/reset,其余属性透传(诊断脚本,不入 src)。"""

    def __init__(self, env: Any, stats: dict[str, float]) -> None:
        self._env = env
        self._stats = stats

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = self._env.step(*args, **kwargs)
        self._stats["env_step"] += time.perf_counter() - start
        return result

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = self._env.reset(*args, **kwargs)
        self._stats["env_reset"] += time.perf_counter() - start
        return result


def _install_probes(stats: dict[str, float], eval_ends: list[float]) -> None:
    """在本进程内包 runner 模块缝(不落盘、不进 src)。"""

    def timed_build_env(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        env = build_routing_env(*args, **kwargs)
        stats["env_build"] += time.perf_counter() - start
        return _TimingEnv(env, stats)

    original_eval = runner_mod.greedy_eval

    def timed_eval(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = original_eval(*args, **kwargs)
        stats["greedy_eval_total"] += time.perf_counter() - start
        eval_ends.append(time.perf_counter())
        return result

    class _TimingPPO(PPO):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            start = time.perf_counter()
            super().__init__(*args, **kwargs)
            stats["model_build"] += time.perf_counter() - start

        def learn(self, *args: Any, **kwargs: Any) -> Any:
            stats["learn_start_abs"] = time.perf_counter()
            start = time.perf_counter()
            try:
                return super().learn(*args, **kwargs)
            finally:
                stats["learn_total"] += time.perf_counter() - start
                stats["learn_end_abs"] = time.perf_counter()

        def train(self) -> None:
            start = time.perf_counter()
            try:
                super().train()
            finally:
                stats["ppo_train"] += time.perf_counter() - start

    runner_mod.build_routing_env = timed_build_env
    runner_mod.greedy_eval = timed_eval
    runner_mod.PPO = _TimingPPO


def _gpu_sampler(stop: threading.Event, out_path: Path) -> None:
    """1Hz 轮询 nvidia-smi(全部 GPU)写 CSV;t,index,util_pct,mem_mib。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("t,gpu_index,util_gpu_pct,mem_used_mib\n")
        start = time.perf_counter()
        while not stop.is_set():
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False, timeout=10,
            )
            now = time.perf_counter() - start
            for line in proc.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 3:
                    handle.write(f"{now:.2f},{parts[0]},{parts[1]},{parts[2]}\n")
            handle.flush()
            stop.wait(1.0)


def _scratch_config(tag: str) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    root = (OUT_DIR / "scratch" / "single" / tag).resolve()
    cfg["paths"]["runs_dir"] = str(root / "runs")
    cfg["paths"]["ckpts_dir"] = str(root / "ckpts")
    return cfg


def run_profiled(device: str, episodes: int, rep: int) -> None:
    stats: dict[str, float] = {key: 0.0 for key in (
        "env_build", "model_build", "env_step", "env_reset",
        "greedy_eval_total", "learn_total", "ppo_train", "learn_start_abs", "learn_end_abs",
    )}
    eval_ends: list[float] = []
    _install_probes(stats, eval_ends)
    tag = f"{device}_rep{rep}"
    cfg = _scratch_config(tag)
    stop = threading.Event()
    sampler = threading.Thread(target=_gpu_sampler, args=(stop, OUT_DIR / f"gpu_ts_{tag}.csv"), daemon=True)
    sampler.start()
    wall_start = time.perf_counter()
    run_pretrain(seed=SEED, episodes=episodes, device=device, config=cfg)
    wall = time.perf_counter() - wall_start
    stop.set()
    sampler.join(timeout=5)
    # 相位分解:periodic eval = learn 窗口内结束的 greedy_eval;final = learn 后那次
    learn_end = stats["learn_end_abs"]
    periodic_eval = sum(t <= learn_end for t in eval_ends)
    final_eval = len(eval_ends) - periodic_eval
    setup = stats["learn_start_abs"] - wall_start if stats["learn_start_abs"] else 0.0
    summary = {
        "device": device, "episodes": episodes, "seed": SEED, "wall_s": round(wall, 3),
        "eps_per_s": round(episodes / wall, 4),
        "phases_s": {key: round(value, 3) for key, value in stats.items() if not key.endswith("_abs")},
        "derived_s": {
            "setup_env_model": round(setup, 3),
            "post_learn_eval_and_io": round(wall - setup - stats["learn_total"], 3),
            "rollout_excl_train_eval": round(stats["learn_total"] - stats["ppo_train"] - stats["greedy_eval_total"], 3),
            "env_step_share_of_learn_pct": round(100 * (stats["env_step"] + stats["env_reset"]) / stats["learn_total"], 1),
            "greedy_eval_calls_periodic": periodic_eval,
            "greedy_eval_calls_final": final_eval,
        },
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (OUT_DIR / f"phases_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_profiler_segment(device: str) -> None:
    """torch.profiler 归因短段:2 个 rollout(n_steps=50×N=41→2 次 PPO 更新)。"""
    from torch.profiler import ProfilerActivity, profile

    cfg = _scratch_config("profsegment")
    spec = runner_mod._RunSpec(
        arm="pretrain", topo=cfg["pretrain"]["topology"], rate=int(cfg["pretrain"]["avgrate"]),
        seed=SEED, pbrs=True, freeze=False, source_run_id=None, episodes=2,
        out_root=None, ckpts_dir=OUT_DIR / "scratch" / "profsegment" / "ckpts",
        device=device, config=cfg,
    )
    env = build_routing_env(spec.topo, spec.rate, pbrs=True, seed=spec.seed, config=cfg)
    vec = VecMonitor(NodeFanVecEnv(env))
    model = runner_mod._build_model(vec, spec, spec.ckpts_dir)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    steps_per_episode = int(cfg["env"]["episode_steps"])
    with profile(activities=activities) as prof:
        model.learn(total_timesteps=2 * steps_per_episode * vec.num_envs, progress_bar=False)
    lines = [f"# torch.profiler segment: device={device}, 2 rollouts (2 PPO updates), seed={SEED}"]
    lines.append("\n## sorted by self_cpu_time_total\n")
    lines.append(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=35))
    if device == "cuda":
        lines.append("\n## sorted by self_cuda_time_total\n")
        lines.append(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    (OUT_DIR / f"profiler_{device}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"profiler table -> profiling/profiler_{device}.txt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--profiler-segment", action="store_true")
    args = parser.parse_args()
    if args.profiler_segment:
        run_profiler_segment(args.device)
    else:
        run_profiled(args.device, args.episodes, args.rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
