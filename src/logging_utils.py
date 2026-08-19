import csv
import json
import logging
import os
import sys
import time


def make_run_id(job):
    parts = [job["mode"], job["scenario"], job.get("target_topo") or job["topo"], job["method"]]
    if job.get("variant"):
        parts.append(job["variant"])
    if job.get("rate") is not None:
        parts.append("r%d" % job["rate"])
    if job.get("fail_ratio"):
        parts.append("f%d" % int(job["fail_ratio"] * 100))
    if job.get("k_paths"):
        parts.append("k%d" % job["k_paths"])
    parts.append("seed%d" % job["seed"])
    return "_".join(str(p) for p in parts)


def setup_logger(run_dir, run_id):
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_manifest(run_dir, job, cfg, status):
    manifest = {"job": job, "status": status, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        json.dump({"job": job, "config": cfg}, f, indent=2, default=str)


class MetricsWriter:
    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, "metrics.csv")
        self.fields = ["episode", "phase", "reward", "delay", "loss", "actor_loss",
                       "critic_loss", "eval_delay", "eval_loss", "elapsed_s"]

    def write(self, row):
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: "" if row.get(k) is None else row.get(k) for k in self.fields})


def mark_done(run_dir, ok=True):
    with open(os.path.join(run_dir, "DONE" if ok else "FAILED"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
