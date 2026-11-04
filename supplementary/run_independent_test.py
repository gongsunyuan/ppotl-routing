"""Independent evaluation of existing best policies; no training or model updates."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces

from trl_sb3.common.config import load_config
from trl_sb3.common.envs import build_routing_env
from trl_sb3.eval.evaluate import EVAL_AGG_KEYS, greedy_eval
from trl_sb3.eval.heuristics import HEURISTIC_SPLIT_FNS, rollout_split_episode
from trl_sb3.eval.ospf import ospf_policy
from trl_sb3.policy.policy import ActorCriticPolicy

TEST_SEEDS = list(range(20000, 20010))
ARMS = ("A1", "A1b", "A2", "A3", "A3b", "A4", "A5", "A6")
SCENARIOS = {
    "lf": ("Claranet_failure.gml", 500),
    "abilene": ("Abilene.gml", 500),
    "nsfcnet": ("NFSCNET.gml", 500),
    "surge": ("Claranet.gml", 1500),
}


def load_best_policy(weights: Path, device: str) -> ActorCriticPolicy:
    policy = ActorCriticPolicy(
        spaces.Box(-np.inf, np.inf, (78,), dtype=np.float64),
        spaces.Discrete(3),
        lambda _: 1e-5,
        net_layers=(99, 128, 64),
    )
    policy.load_state_dict(
        th.load(weights, weights_only=True, map_location=device), strict=True
    )
    policy.to(device)
    policy.eval()
    return policy


def policy_fn(policy: ActorCriticPolicy):
    def predict(obs: np.ndarray) -> np.ndarray:
        actions, _ = policy.predict(obs, deterministic=True)
        return np.asarray(actions, dtype=np.int64)

    return predict


def heuristic_eval(
    arm: str, topo: str, rate: int, config: dict
) -> dict:
    env = build_routing_env(
        topo, rate, pbrs=False, seed=TEST_SEEDS[0], config=config
    )
    episodes = []
    for seed in TEST_SEEDS:
        env.reset(seed=seed)
        episode = rollout_split_episode(
            env, HEURISTIC_SPLIT_FNS[arm], int(config["env"]["episode_steps"])
        )
        episodes.append({"seed": seed, **episode})
    scores = [step["r_mean"] for ep in episodes for step in ep["steps"]]
    return {
        "episodes": episodes,
        "r_mean_mean": float(np.mean(scores)),
        "rd_mean": float(np.mean([ep["rd_mean"] for ep in episodes])),
        "rp_mean": float(np.mean([ep["rp_mean"] for ep in episodes])),
        "th_mean": float(np.mean([ep["th_mean"] for ep in episodes])),
    }


def summary_row(
    manifest: dict,
    original: dict,
    result: dict,
    policy_path: Path,
) -> dict:
    best = original.get("best") or {}
    last = original.get("final_last") or {}
    return {
        "run_id": manifest["run_id"],
        "arm": manifest["arm"],
        "topo": manifest["topo"],
        "rate": manifest["rate"],
        "train_seed": manifest["seed"],
        "source_run_id": manifest.get("source_run_id") or "",
        "selected_episode": best.get("episode", ""),
        "selection_score": best.get("r_mean_mean", ""),
        "original_best_score": original["final"]["r_mean_mean"],
        "original_final_last_score": last.get("r_mean_mean", ""),
        "independent_test_score": result["r_mean_mean"],
        "independent_rd_mean": result["rd_mean"],
        "independent_rp_mean": result["rp_mean"],
        "independent_th_mean": result["th_mean"],
        "n_test_episodes": len(TEST_SEEDS),
        "test_seed_first": TEST_SEEDS[0],
        "test_seed_last": TEST_SEEDS[-1],
        "policy_path": str(policy_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    th.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    out_root = root / "revision_independent_test" / "results" / args.scenario
    out_root.mkdir(parents=True, exist_ok=True)
    topo, rate = SCENARIOS[args.scenario]
    protocol = {
        "protocol": "revision-best-checkpoint-independent-test-v1",
        "test_seeds": TEST_SEEDS,
        "selection_seeds_excluded": list(range(10000, 10010)),
        "deterministic_actions": True,
        "pbrs_during_evaluation": False,
        "training_performed": False,
        "scenario": args.scenario,
        "topology": topo,
        "rate": rate,
        "device": args.device,
    }
    protocol_path = out_root / "protocol.json"
    if protocol_path.exists() and not args.force:
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old["test_seeds"] != TEST_SEEDS:
            raise RuntimeError("Existing output uses different test seeds")
    protocol_path.write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    tasks = []
    for directory in sorted((root / "runs").iterdir()):
        if not directory.is_dir() or not (directory / "DONE").exists():
            continue
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["topo"], manifest["rate"]) != (topo, rate):
            continue
        arm = manifest["arm"]
        if arm not in (*ARMS, "A0", "OSPF", "ECMP", "LB", "RR"):
            continue
        original = json.loads((directory / "eval.json").read_text(encoding="utf-8"))
        if arm in ARMS:
            weights = directory / "best_policy.pt"
        elif arm == "A0":
            weights = root / "runs" / manifest["source_run_id"] / "best_policy.pt"
        else:
            weights = directory
        if arm in (*ARMS, "A0") and not weights.exists():
            raise FileNotFoundError(weights)
        tasks.append((arm, manifest["seed"], directory, manifest, original, weights))

    order = {arm: i for i, arm in enumerate((*ARMS, "A0", "OSPF", "ECMP", "LB", "RR"))}
    tasks.sort(key=lambda item: (order[item[0]], item[1]))
    rows = []
    started = time.time()
    for index, (arm, seed, directory, manifest, original, weights) in enumerate(tasks, 1):
        raw_path = out_root / f"{arm}_seed{seed}.json"
        if raw_path.exists() and not args.force:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            if payload["protocol"]["test_seeds"] != TEST_SEEDS:
                raise RuntimeError(f"Seed mismatch in {raw_path}")
        else:
            if arm in (*ARMS, "A0"):
                policy = load_best_policy(weights, args.device)
                result = greedy_eval(
                    policy_fn(policy), topo, rate, eval_seeds=TEST_SEEDS,
                    config=manifest["config"],
                )
                del policy
            elif arm == "OSPF":
                result = greedy_eval(
                    ospf_policy, topo, rate, eval_seeds=TEST_SEEDS,
                    config=manifest["config"],
                )
            else:
                result = heuristic_eval(arm, topo, rate, manifest["config"])
            payload = {
                "protocol": protocol,
                "summary": summary_row(manifest, original, result, weights),
                "evaluation": result,
            }
            raw_path.write_text(json.dumps(payload), encoding="utf-8")
        rows.append(payload["summary"])
        print(
            f"[{args.scenario} {index}/{len(tasks)}] {arm} seed={seed} "
            f"test={payload['summary']['independent_test_score']:.9f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    fields = list(rows[0])
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"DONE {args.scenario}: {len(rows)} evaluations", flush=True)


if __name__ == "__main__":
    main()
