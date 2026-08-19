"""Shared job building and environment-parameter logic for CPU and GPU orchestrators."""

ALL_METHODS = ["ospf", "ppo_vanilla", "ppo_naive", "dqn", "grlps", "ppotl"]
ABLATION_METHODS = ["ppotl", "ppotl_freeze_only", "ppotl_pbrs_only", "ppo_naive", "ppo_vanilla"]
PPO_METHODS = {"ppo_vanilla", "ppo_naive", "ppotl", "ppotl_freeze_only", "ppotl_pbrs_only", "ppo_local"}


def method_flags(method):
    return {
        "load_ckpt": method in {"ppo_naive", "ppotl", "ppotl_freeze_only", "ppotl_pbrs_only", "grlps"},
        "needs_pretrain": method in {"ppo_naive", "ppotl", "ppotl_freeze_only", "ppotl_pbrs_only", "grlps"},
        "adapts": method != "ospf",
    }


def pretrain_key(method, topo, seed, k):
    fam = "grlps" if method == "grlps" else "ppo"
    return "%s_%s_seed%d_k%s" % (fam, topo, seed, k)


def group_key(job):
    parts = [job["scenario"], job.get("target_topo") or job["topo"], job["method"],
             str(job.get("rate")), str(job.get("fail_ratio")), str(job.get("variant")),
             str(job.get("k_paths")), job.get("traffic_mode", "constant"),
             str(job.get("dynamic_failures", False))]
    return "|".join(parts)


def build_jobs(mode, cfg, scenarios, seeds, topo_filter):
    mc = cfg[mode]
    topologies = [t for t in mc["topologies"] if not topo_filter or topo_filter in t]
    seeds = seeds or mc["seeds"]
    jobs = []

    def add(**kw):
        jobs.append({"mode": mode, **kw})

    if "S1" in scenarios:
        for topo in topologies:
            for rate in mc["constant_rates"]:
                for seed in seeds:
                    for method in ALL_METHODS:
                        add(scenario="S1", topo=topo, method=method, rate=rate,
                            traffic_mode="constant", seed=seed)

    if "S2" in scenarios:
        for topo in topologies:
            for seed in seeds:
                for method in ALL_METHODS:
                    add(scenario="S2", topo=topo, method=method, traffic_mode="varying",
                        seed=seed, eval_rate_sweep=True)

    if "S3" in scenarios:
        for topo in topologies:
            for fr in mc["fail_ratios"]:
                for seed in seeds:
                    for method in ALL_METHODS:
                        add(scenario="S3", topo=topo, method=method, fail_ratio=fr,
                            traffic_mode="varying", dynamic_failures=True, seed=seed)

    if "S4" in scenarios:
        src = "cernet"
        for tgt in [t for t in mc["topologies"] if t != src]:
            if topo_filter and topo_filter not in tgt:
                continue
            for seed in seeds:
                for method in ALL_METHODS:
                    add(scenario="S4", topo=src, target_topo=tgt, method=method, seed=seed,
                        traffic_mode="varying")
                add(scenario="S4", topo=tgt, method="ppo_local", seed=seed, traffic_mode="varying")

    if "A" in scenarios:
        for topo in [t for t in ("cernet", "abilene") if not topo_filter or topo_filter in t]:
            for seed in seeds:
                for method in ABLATION_METHODS:
                    add(scenario="A", topo=topo, method=method, traffic_mode="varying", seed=seed)

    if "H" in scenarios and mc.get("sensitivity"):
        topo = cfg["sensitivity"]["topo"]
        if topo_filter and topo_filter not in topo:
            return jobs
        for seed in seeds:
            for k in cfg["sensitivity"]["k_paths"]:
                add(scenario="H", topo=topo, method="ppotl", traffic_mode="varying",
                    seed=seed, k_paths=k, variant="k%d" % k)
            for fr in cfg["sensitivity"]["freeze_fraction"]:
                add(scenario="H", topo=topo, method="ppotl", traffic_mode="varying",
                    seed=seed, freeze_fraction=fr, variant="freeze%.2f" % fr)
            for z in cfg["sensitivity"]["zeta1"]:
                add(scenario="H", topo=topo, method="ppotl", traffic_mode="varying",
                    seed=seed, zeta1=z, variant="zeta%.1f" % z)
            for a in cfg["sensitivity"]["alpha1"]:
                add(scenario="H", topo=topo, method="ppotl", traffic_mode="varying",
                    seed=seed, alpha1=a, variant="alpha%.1f" % a)
    return jobs


def env_params_from_job(job, cfg, phase):
    """Common env construction parameters, shared by CPU and GPU backends."""
    traffic_mode = job.get("traffic_mode", "constant")
    rate = job.get("rate", cfg["traffic"]["source_mean_rate"]) if phase == "target" else cfg["traffic"]["source_mean_rate"]
    if traffic_mode == "varying":
        rate = cfg["traffic"]["source_mean_rate"]
    topo = job.get("target_topo") or job["topo"]
    return {
        "topo": topo,
        "traffic_mode": traffic_mode,
        "rate": rate,
        "fail_ratio": job.get("fail_ratio", 0.0) if phase == "target" else 0.0,
        "dynamic_failures": bool(job.get("dynamic_failures", False)) and phase == "target",
        "k_paths": job.get("k_paths"),
        "alpha1": job.get("alpha1"),
    }


TOPO_COST = {"nsfcnet": 1.0, "abilene": 1.2, "claranet": 1.5, "cernet": 4.0}


def job_cost(job, cfg):
    mc = cfg[job["mode"]]
    topo = job.get("target_topo") or job["topo"]
    c = TOPO_COST.get(topo, 2.0) * mc["adapt_episodes"]
    flags = method_flags(job["method"])
    if flags["needs_pretrain"]:
        c += TOPO_COST.get(job["topo"], 2.0) * mc["pretrain_episodes"]
    return c
