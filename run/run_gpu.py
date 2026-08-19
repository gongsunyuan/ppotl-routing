"""Multi-GPU orchestrator.

- Auto-detects available GPUs (name, free/total memory) and picks sensible defaults.
- Groups PPO-family jobs into stacked-seed ensemble units (L3) when --ensemble (default).
- Shards units across (GPU x proc) workers by cost, spawns one process per worker.
- Fully resumable: deterministic run ids + DONE markers, shared with the CPU backend.

Usage:
    python run/run_gpu.py --mode full                  # all GPUs, defaults
    python run/run_gpu.py --mode full --gpus 0,1 --procs-per-gpu 2
    python run/run_gpu.py --mode smoke --no-ensemble   # quick single-seed check
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

import torch
import torch.multiprocessing as mp

from src.jobs import build_jobs, group_key, job_cost, method_flags, pretrain_key, PPO_METHODS
from src.logging_utils import make_run_id

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg():
    with open(os.path.join(ROOT, "config", "default.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_gpus(requested=None):
    if not torch.cuda.is_available():
        return []
    infos = []
    n = torch.cuda.device_count()
    for i in range(n):
        if requested and i not in requested:
            continue
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        infos.append({"idx": i, "name": props.name, "total_gb": total / 2 ** 30,
                      "free_gb": free / 2 ** 30})
    return infos


def auto_n_vec_envs(infos):
    free = min(g["free_gb"] for g in infos) if infos else 0
    if free >= 16:
        return 512
    if free >= 8:
        return 256
    return 128


def build_units(jobs, cfg, ensemble):
    """Group jobs into executable units."""
    units = []
    if ensemble:
        groups = {}
        rest = []
        for j in jobs:
            if j["method"] in PPO_METHODS:
                groups.setdefault(group_key(j), []).append(j)
            else:
                rest.append(j)
        for key, gj in groups.items():
            gj = sorted(gj, key=lambda x: x["seed"])
            cost = sum(job_cost(j, cfg) for j in gj) * 0.6
            units.append({"kind": "ensemble_ppo", "jobs": gj, "cost": cost})
        jobs = rest
    for j in jobs:
        units.append({"kind": "single", "jobs": [j], "cost": job_cost(j, cfg)})
    return units


def unit_run_dirs(unit, mode):
    return [os.path.join(ROOT, "logs", mode, make_run_id(j)) for j in unit["jobs"]]


def unit_done(unit, mode):
    return all(os.path.exists(os.path.join(d, "DONE")) for d in unit_run_dirs(unit, mode))


def shard_units(units, n_workers):
    """Greedy least-loaded assignment by cost (heaviest first)."""
    order = sorted(range(len(units)), key=lambda i: -units[i]["cost"])
    loads = [0.0] * n_workers
    assign = [[] for _ in range(n_workers)]
    for i in order:
        w = min(range(n_workers), key=lambda k: loads[k])
        assign[w].append(i)
        loads[w] += units[i]["cost"]
    return assign, loads


def _pretrain_dir(mode, key):
    return os.path.join(ROOT, "logs", mode, "gpre_" + key)


def _pretrain_ready(pdir):
    return os.path.exists(os.path.join(pdir, "DONE")) and \
        os.path.exists(os.path.join(pdir, "ckpt.pt"))


def ensure_pretrain(pdir, args, build_fn):
    """Idempotent, concurrency-safe pretrain: mkdir-based lock + DONE double-check."""
    if _pretrain_ready(pdir) and not args.force:
        return os.path.join(pdir, "ckpt.pt")
    lock = pdir + ".lock"
    os.makedirs(pdir, exist_ok=True)
    deadline = time.time() + 3600
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            if _pretrain_ready(pdir) and not args.force:
                return os.path.join(pdir, "ckpt.pt")
            if time.time() > deadline:
                print("WARNING: stale pretrain lock %s, taking over" % lock, flush=True)
                try:
                    os.rmdir(lock)
                except OSError:
                    pass
                continue
            time.sleep(2)
    try:
        if _pretrain_ready(pdir) and not args.force:
            return os.path.join(pdir, "ckpt.pt")
        return build_fn()
    finally:
        try:
            os.rmdir(lock)
        except OSError:
            pass


def worker(rank, world):
    gpu_idx, proc_idx, unit_ids, units, cfg, args, mode = world[rank]
    device = "cuda:%d" % gpu_idx if gpu_idx >= 0 else "cpu"
    if gpu_idx >= 0:
        torch.cuda.set_device(gpu_idx)
        if cfg["gpu"].get("tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
    tag = "[gpu%d p%d]" % (gpu_idx, proc_idx)
    print("%s started, %d units" % (tag, len(unit_ids)), flush=True)
    gcfg = cfg["gpu"]
    for n, i in enumerate(unit_ids):
        u = units[i]
        if unit_done(u, mode) and not args.force:
            print("%s skip done unit %s" % (tag, unit_name(u)), flush=True)
            continue
        t0 = time.time()
        try:
            execute_unit(u, cfg, device, args, tag)
            print("%s [%d/%d] %s OK %.1fs" % (tag, n + 1, len(unit_ids), unit_name(u), time.time() - t0), flush=True)
        except Exception:
            print("%s [%d/%d] %s FAILED\n%s" % (tag, n + 1, len(unit_ids), unit_name(u),
                                                traceback.format_exc()), flush=True)
            for d in unit_run_dirs(u, mode):
                try:
                    with open(os.path.join(d, "FAILED"), "w") as f:
                        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
                except OSError:
                    pass
    print("%s finished" % tag, flush=True)


def unit_name(u):
    j = u["jobs"][0]
    seeds = ",".join(str(x["seed"]) for x in u["jobs"])
    base = make_run_id(j).rsplit("_seed", 1)[0]
    return "%s#%s(%d)" % (u["kind"], base, len(u["jobs"]))


def execute_unit(u, cfg, device, args, tag):
    from src.train_gpu import run_adapt_single, run_adapt_ensemble, run_pretrain_gpu, run_pretrain_grlps_gpu
    mode = args.mode
    gcfg = cfg["gpu"]
    n_vec = args.n_vec_envs or gcfg["n_vec_envs"]

    if u["kind"] == "ensemble_ppo":
        jobs = u["jobs"]
        seeds = [j["seed"] for j in jobs]
        j0 = jobs[0]
        fam = "ppo"
        k = j0.get("k_paths") or cfg["paper_defaults"]["k_paths"]
        key = "%s_%s_seeds-%s_k%s" % (fam, j0["topo"], "-".join(map(str, sorted(seeds))), k)
        pdir = _pretrain_dir(mode, key)
        flags = method_flags("ppo_vanilla" if j0["method"] == "ppo_local" else j0["method"])
        ckpt = None
        if flags["needs_pretrain"]:
            pj = dict(j0)
            pj["scenario"] = "P"
            n_copies = len(seeds) * max(8, n_vec // len(seeds))
            ckpt = ensure_pretrain(pdir, args, lambda: run_pretrain_gpu(
                pj, cfg, pdir, device, seeds, n_copies, fam)[0])
        if flags["load_ckpt"]:
            run_adapt_ensemble(jobs, cfg, device, ckpt, gcfg)
        else:
            run_adapt_ensemble(jobs, cfg, device, None, gcfg)
        return

    job = u["jobs"][0]
    method = job["method"]
    flags = method_flags("ppo_vanilla" if method == "ppo_local" else method)
    k = job.get("k_paths") or cfg["paper_defaults"]["k_paths"]
    if flags["needs_pretrain"]:
        fam = "grlps" if method == "grlps" else "ppo"
        key = pretrain_key(method, job["topo"], job["seed"], k)
        pdir = _pretrain_dir(mode, key)
        pj = dict(job)
        pj["scenario"] = "P"
        if fam == "grlps":
            ckpt = ensure_pretrain(pdir, args, lambda: run_pretrain_grlps_gpu(
                pj, cfg, pdir, device, job["seed"], max(16, n_vec // 8))[0])
        else:
            ckpt = ensure_pretrain(pdir, args, lambda: run_pretrain_gpu(
                pj, cfg, pdir, device, [job["seed"]], max(16, n_vec // 8), fam)[0])
    else:
        ckpt = None
    run_dir = os.path.join(ROOT, "logs", mode, make_run_id(job))
    os.makedirs(run_dir, exist_ok=True)
    run_adapt_single(job, cfg, run_dir, device, ckpt=ckpt if flags["load_ckpt"] else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--scenarios", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--topo", default=None)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--gpus", default=None, help="comma list of GPU ids, default: all")
    ap.add_argument("--procs-per-gpu", type=int, default=None)
    ap.add_argument("--n-vec-envs", type=int, default=None)
    ap.add_argument("--ensemble/--no-ensemble", dest="ensemble", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    requested = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    infos = detect_gpus(requested)
    if not infos:
        print("WARNING: no CUDA GPUs visible (CPU-only torch build or none free).")
        print("Falling back to the CPU device with the *vectorized* backend; install a CUDA")
        print("torch build on the training machine to use real GPUs automatically.")
        infos = [{"idx": -1, "name": "CPU-fallback (vectorized)", "total_gb": 0.0, "free_gb": 0.0}]
    print("Detected GPUs:")
    for g in infos:
        print("  [%d] %-28s %.1f GB free / %.1f GB" % (g["idx"], g["name"], g["free_gb"], g["total_gb"]))

    if args.n_vec_envs is None and not cfg["gpu"].get("n_vec_envs"):
        cfg["gpu"]["n_vec_envs"] = auto_n_vec_envs(infos)
    n_vec = args.n_vec_envs or cfg["gpu"]["n_vec_envs"]
    ppg = args.procs_per_gpu if args.procs_per_gpu is not None else cfg["gpu"]["procs_per_gpu"]
    if infos[0]["idx"] < 0:
        if args.procs_per_gpu is None:
            ppg = 1  # CPU fallback: single process unless explicitly requested
        if args.n_vec_envs is None and not cfg["gpu"].get("n_vec_envs"):
            cfg["gpu"]["n_vec_envs"] = 128
    ensemble = args.ensemble if args.ensemble is not None else bool(cfg["gpu"]["ensemble"])
    cfg["gpu"]["n_vec_envs"] = n_vec
    print("config: n_vec_envs=%d procs/gpu=%d ensemble=%s" % (n_vec, ppg, ensemble))

    scenarios = args.scenarios.split(",") if args.scenarios else cfg[args.mode]["scenarios"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    jobs = build_jobs(args.mode, cfg, scenarios, seeds, args.topo)
    if args.filter:
        jobs = [j for j in jobs if args.filter in make_run_id(j)]
    units = build_units(jobs, cfg, ensemble)
    todo = [u for u in units if not unit_done(u, args.mode) or args.force]
    skipped = len(units) - len(todo)
    print("planned: %d jobs -> %d units (%d already done)" % (len(jobs), len(units), skipped))
    if args.dry_run:
        for u in sorted(todo, key=lambda x: -x["cost"]):
            print("DRY %-8s cost=%8.0f %s" % (u["kind"], u["cost"], unit_name(u)))
        return

    n_workers = len(infos) * ppg
    assign, loads = shard_units(todo, n_workers)
    gpu_ids = [g["idx"] for g in infos]
    world = [(gpu_ids[w // ppg], w % ppg, assign[w], todo, cfg, args, args.mode)
             for w in range(n_workers)]
    print("workers: %d (gpus %s, %d proc/gpu), loads: %s" %
          (n_workers, gpu_ids, ppg, ["%.0f" % l for l in loads]))
    if n_workers == 1:
        worker(0, world)
        return
    mp.spawn(worker, nprocs=n_workers, args=(world,))


if __name__ == "__main__":
    main()
