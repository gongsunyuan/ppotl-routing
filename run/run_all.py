import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.train import run_pretrain, run_adapt
from src.jobs import build_jobs, method_flags, pretrain_key
from src.logging_utils import make_run_id, setup_logger, write_manifest, mark_done

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg():
    with open(os.path.join(ROOT, "config", "default.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dir_for(mode, run_id):
    d = os.path.join(ROOT, "logs", mode, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def execute(jobs, cfg, force=False, dry=False):
    pretrains = {}
    stats = {"done": 0, "skipped": 0, "failed": 0}
    todo = []
    for job in jobs:
        rid = make_run_id(job)
        d = os.path.join(ROOT, "logs", job["mode"], rid)
        if os.path.exists(os.path.join(d, "DONE")) and not force:
            stats["skipped"] += 1
            continue
        todo.append((rid, d, job))
    if dry:
        for rid, d, job in todo:
            print("DRY", rid)
        print("total=%d done_before=%d" % (len(todo), stats["skipped"]))
        return stats
    print("jobs to run: %d (skipped %d already done)" % (len(todo), stats["skipped"]))
    t_start = time.time()
    for i, (rid, d, job) in enumerate(todo):
        os.makedirs(d, exist_ok=True)
        logger = setup_logger(d, rid)
        logger.info("=== start %s (%d/%d) ===" % (rid, i + 1, len(todo)))
        try:
            ckpt = None
            flags = method_flags(job["method"])
            key = None
            if job["method"] == "ppo_local":
                flags = method_flags("ppo_vanilla")
            if flags["needs_pretrain"]:
                key = pretrain_key(job["method"], job["topo"], job["seed"],
                                   job.get("k_paths", cfg["paper_defaults"]["k_paths"]))
                if key in pretrains:
                    ckpt = pretrains[key]
                else:
                    prid = "pretrain_" + key
                    pd = run_dir_for(job["mode"], prid)
                    if os.path.exists(os.path.join(pd, "DONE")) and os.path.exists(os.path.join(pd, "ckpt.pt")) and not force:
                        ckpt = os.path.join(pd, "ckpt.pt")
                    else:
                        pjob = dict(job)
                        pjob["scenario"] = "P"
                        ckpt, _ = run_pretrain(pjob, cfg, pd)
                    pretrains[key] = ckpt
            logger.info("using pretrain ckpt=%s" % ckpt)
            run_adapt(job, cfg, d, ckpt=ckpt)
            stats["done"] += 1
        except Exception:
            stats["failed"] += 1
            logger.error(traceback.format_exc())
            write_manifest(d, job, cfg, "failed")
            mark_done(d, ok=False)
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(todo) - i - 1)
        print("[%d/%d] %s done | failed=%d | ETA %.1f min" %
              (i + 1, len(todo), rid, stats["failed"], eta / 60.0), flush=True)
    print("SUMMARY", stats)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--scenarios", default=None, help="comma list: S1,S2,S3,S4,A,H")
    ap.add_argument("--seeds", default=None, help="comma list of ints, overrides config")
    ap.add_argument("--topo", default=None, help="substring filter on topology")
    ap.add_argument("--filter", default=None, help="substring filter on run description")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    scenarios = args.scenarios.split(",") if args.scenarios else cfg[args.mode]["scenarios"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    cfg[args.mode]["horizon"] = cfg[args.mode]["horizon"]
    jobs = build_jobs(args.mode, cfg, scenarios, seeds, args.topo)
    if args.filter:
        jobs = [j for j in jobs if args.filter in make_run_id(j)]
    print("planned jobs: %d | mode=%s scenarios=%s" % (len(jobs), args.mode, scenarios))
    execute(jobs, cfg, force=args.force, dry=args.dry_run)


if __name__ == "__main__":
    main()
