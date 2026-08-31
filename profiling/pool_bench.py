"""并发池基准(Phase-0 §2.6):子进程 sweep worker(与 tune.py/_execute_trial 同款命令)
的有界并发扫描 + 线程配比测试。

worker 命令 = [sys.executable, -m trl_sb3.run, sweep, --grid <临时网格>, --device D]
——与 tune.py 逐试验子进程完全同构;每任务一个独立网格(pretrain seeds=[s_i] 单种子,
episodes=冒烟预算),同 level 内共享 runs 根但 run_id 互异(种子互异)——这正是主表
--filter 多进程的并发模型(不同 filter 子串 → 不相交任务集,目录幂等)。

每任务用 /usr/bin/time -v 抓 per-process user CPU / maxRSS / CPU%;结果追加
profiling/pool_results.csv。level 间用不同 scratch 根(测量互不污染)。

用法(工作目录 = 仓库根):
  .venv/bin/python profiling/pool_bench.py --device cpu --levels 1,2,4,8,16 --tag conc
  .venv/bin/python profiling/pool_bench.py --device cpu --levels 8 --threads 1 --tag thr1
  CUDA_VISIBLE_DEVICES 未动,--cuda-gpus 显式指定 worker 的 GPU 钉卡(轮转)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import re
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "profiling"
RESULTS_CSV = OUT_DIR / "pool_results.csv"
SEED_BASE = 20_000  # 画像专用种子区,不与调参/试点流 {0..9,100..102} 相交

_TIME_PATTERNS = {
    "user_s": r"User time \(seconds\): ([0-9.]+)",
    "sys_s": r"System time \(seconds\): ([0-9.]+)",
    "cpu_pct": r"Percent of CPU this job got: ([0-9]+)%",
    "max_rss_kib": r"Maximum resident set size \(kbytes\): ([0-9]+)",
    "exit_status": r"Exit status: ([0-9]+)",
}


def _parse_time_file(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed: dict[str, float] = {}
    for key, pattern in _TIME_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def _worker_env(threads: str, cuda_gpu: int | None) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if threads == "default":
            env.pop(key, None)
        else:
            env[key] = threads
    if cuda_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_gpu)
    return env


def _write_grid(path: Path, seed: int, episodes: int, runs_dir: Path, ckpts_dir: Path) -> None:
    grid = {
        "pretrain": {"seeds": [seed], "episodes": episodes},
        "config_overlay": {"paths": {"runs_dir": str(runs_dir), "ckpts_dir": str(ckpts_dir)}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(grid, sort_keys=False), encoding="utf-8")


def _gpu_sampler(stop: threading.Event, out_path: Path) -> None:
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


def run_level(
    *, device: str, level: int, tasks: int, episodes: int, threads: str,
    tag: str, cuda_gpus: list[int] | None, sample_gpu: bool,
) -> dict[str, object]:
    level_root = OUT_DIR / "scratch" / "pool" / tag / f"N{level}"
    runs_dir, ckpts_dir = level_root / "runs", level_root / "ckpts"
    commands: list[list[str]] = []
    # crc32 确定性派生(Python hash 有进程盐,不可用作种子):level 内种子互异、跨 tag/level 不撞
    seed = SEED_BASE + zlib.crc32(f"{tag}-N{level}".encode()) % 1000
    for i in range(tasks):
        grid_path = level_root / f"grid_task{i}.yaml"
        _write_grid(grid_path, seed + i, episodes, runs_dir, ckpts_dir)
        time_out = level_root / f"time_task{i}.txt"
        commands.append([
            "/usr/bin/time", "-v", "-o", str(time_out),
            sys.executable, "-m", "trl_sb3.run", "sweep",
            "--grid", str(grid_path), "--device", device,
        ])

    def launch(index: int, command: list[str]) -> int:
        gpu = cuda_gpus[index % len(cuda_gpus)] if cuda_gpus else None
        return subprocess.run(command, check=False, env=_worker_env(threads, gpu)).returncode

    stop = threading.Event()
    sampler = None
    if sample_gpu and device == "cuda":
        sampler = threading.Thread(target=_gpu_sampler, args=(stop, OUT_DIR / f"gpu_ts_pool_{tag}_N{level}.csv"), daemon=True)
        sampler.start()
    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
        exit_codes = list(pool.map(launch, range(len(commands)), commands))
    wall = time.perf_counter() - wall_start
    stop.set()
    if sampler is not None:
        sampler.join(timeout=5)
    stats = [_parse_time_file(level_root / f"time_task{i}.txt") for i in range(tasks)]
    users = [s["user_s"] for s in stats if "user_s" in s]
    rss = [s["max_rss_kib"] for s in stats if "max_rss_kib" in s]
    cpus = [s["cpu_pct"] for s in stats if "cpu_pct" in s]
    row: dict[str, object] = {
        "tag": tag, "device": device, "threads": threads, "level": level, "tasks": tasks,
        "episodes": episodes, "wall_s": round(wall, 2),
        "agg_eps_per_s": round(tasks * episodes / wall, 3),
        "per_task_eps_per_s": round(episodes * tasks / wall / level, 3),
        "mean_user_s": round(sum(users) / len(users), 2) if users else -1,
        "mean_sys_rss_mib": round(sum(rss) / len(rss) / 1024, 1) if rss else -1,
        "mean_cpu_pct": round(sum(cpus) / len(cpus), 0) if cpus else -1,
        "nonzero_exits": sum(1 for code in exit_codes if code != 0),
    }
    new_file = not RESULTS_CSV.exists()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(json_line(row))
    return row


def json_line(row: dict[str, object]) -> str:
    import json

    return json.dumps(row, ensure_ascii=False)


def measure_startup(device: str, threads: str) -> None:
    """单 subprocess dry-run 墙钟 = 固定启动开销(interpreter+torch import+plan)。"""
    grid_path = OUT_DIR / "scratch" / "pool" / "startup_grid.yaml"
    _write_grid(grid_path, SEED_BASE, 1, OUT_DIR / "scratch" / "pool" / "startup", OUT_DIR / "scratch" / "pool" / "startup")
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "trl_sb3.run", "sweep", "--grid", str(grid_path), "--device", device, "--dry-run"],
        check=False, env=_worker_env(threads, None), capture_output=True, text=True, timeout=300,
    )
    wall = time.perf_counter() - start
    print(f"startup probe (dry-run subprocess, device={device}, threads={threads}): "
          f"wall={wall:.2f}s exit={proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--levels", default="1,2,4,8,16", help="逗号分隔并发度列表")
    parser.add_argument("--episodes", type=int, default=20, help="每任务预训练回合(冒烟预算)")
    parser.add_argument("--tasks-per-level", type=int, default=0, help="0 = max(2N, 8)")
    parser.add_argument("--threads", default="1", help="worker OMP/MKL 线程数:'default' 不设")
    parser.add_argument("--tag", default="conc", help="测量标签(scratch 子目录 + 结果行)")
    parser.add_argument("--cuda-gpus", default=None, help="逗号分隔 worker 轮转钉卡列表(如 1,2,3)")
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("--startup-probe", action="store_true")
    args = parser.parse_args()
    cuda_gpus = [int(g) for g in args.cuda_gpus.split(",")] if args.cuda_gpus else None
    if args.startup_probe:
        measure_startup(args.device, args.threads)
        return 0
    for level in [int(x) for x in args.levels.split(",")]:
        tasks = args.tasks_per_level or max(2 * level, 8)
        run_level(
            device=args.device, level=level, tasks=tasks, episodes=args.episodes,
            threads=args.threads, tag=args.tag, cuda_gpus=cuda_gpus, sample_gpu=args.sample_gpu,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
