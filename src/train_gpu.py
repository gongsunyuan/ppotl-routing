"""GPU training pipeline: vectorized pretrain/adapt for PPO (ensemble-capable), DQN,
GRL-PS and OSPF. Writes the same per-run artifacts as the CPU backend (metrics.csv,
eval.json, manifest.json, DONE), so make_figures.py aggregates both identically."""
import json
import os
import time

import numpy as np
import torch

from .env.vec_env import VecNetEnv
from .agents.ppo_gpu import PPOGpuAgent
from .agents.dqn_gpu import DQNGpuAgent, GRLPSGpuAgent
from .jobs import env_params_from_job, method_flags, PPO_METHODS
from .logging_utils import MetricsWriter, setup_logger, write_manifest, mark_done
from .evaluate import convergence_episode


def _local_cfg(cfg, mode):
    local = dict(cfg)
    local["horizon"] = cfg[mode]["horizon"]
    return local


def build_vec_env(job, cfg, phase, seed, n_copies, device, fixed_norm=None, train_norm=None):
    p = env_params_from_job(job, cfg, phase)
    local = _local_cfg(cfg, job["mode"])
    if train_norm is None:
        train_norm = phase == "source"
    return VecNetEnv(p["topo"], local, seed=seed, traffic_mode=p["traffic_mode"],
                     rate=p["rate"], fail_ratio=p["fail_ratio"],
                     dynamic_failures=p["dynamic_failures"], train_norm=train_norm,
                     fixed_norm=fixed_norm, k_paths=p["k_paths"], alpha1=p["alpha1"],
                     n_copies=n_copies, device=device)


def ppo_variant_flags(method):
    return {
        "use_pbrs": method in ("ppotl", "ppotl_pbrs_only"),
        "freeze": 1.0 if method in ("ppotl", "ppotl_freeze_only") else 0.0,
    }


# ---------------- PPO rollouts ----------------

def rollout_ppo(agent, env, train=True):
    S, H = agent.S, env.horizon
    s = env.reset()
    N = s.shape[0]
    R = N // S
    dev = s.device
    states = torch.empty(H, N, env.state_dim, device=dev)
    actions = torch.empty(H, N, dtype=torch.long, device=dev)
    logps = torch.empty(H, N, device=dev)
    rewards = torch.empty(H, N, device=dev)
    nexts = torch.empty(H, N, env.state_dim, device=dev)
    ep_reward = torch.zeros(N, device=dev)
    for t in range(H):
        if S == 1:
            a, lp = agent.act(s)
        else:
            a, lp = agent.act_ensemble(s.view(S, R, -1))
            a, lp = a.reshape(N), lp.reshape(N)
        s2, r, done, info = env.step(a)
        r_shaped = agent.shape_reward(r.float(), s, s2) if train else r.float()
        states[t], actions[t], logps[t], rewards[t], nexts[t] = s, a, lp, r_shaped, s2
        ep_reward += r.float()
        s = s2
    if train:
        agent.update(states, actions, logps, rewards, nexts)
    return ep_reward


@torch.no_grad()
def eval_ppo(agent, env, episodes):
    S = agent.S
    H = env.horizon
    N = env.n_copies
    R = N // S
    per_seed_d, per_seed_l, per_seed_r = [], [], []
    for _ in range(episodes):
        s = env.reset()
        for t in range(H):
            if S == 1:
                a = agent.act_greedy(s)
            else:
                a = agent.act_greedy_ensemble(s.view(S, R, -1)).reshape(N)
            s, r, done, info = env.step(a)
        per_seed_d.append(env.last_lat.view(S, R).mean(1).cpu().numpy())
        per_seed_l.append(env.last_los.view(S, R).mean(1).cpu().numpy())
    d = np.mean(np.stack(per_seed_d), axis=0)
    l = np.mean(np.stack(per_seed_l), axis=0)
    return d, l


# ---------------- DQN rollouts ----------------

def rollout_dqn(agent, env, train=True, augment=None):
    H = env.horizon
    s = env.reset()
    N = s.shape[0]
    ep_reward = torch.zeros(N, device=s.device)
    for t in range(H):
        st = augment(s, env.agent_pair_idx) if augment else s
        a = agent.act_batch(st, greedy=not train)
        s2, r, done, info = env.step(a)
        if train:
            st2 = augment(s2, env.agent_pair_idx) if augment else s2
            agent.observe_batch(st, a, r.float(), st2, torch.zeros(N, device=s.device))
        ep_reward += r.float()
        s = s2
    return ep_reward


@torch.no_grad()
def eval_dqn(agent, env, episodes, augment=None):
    H = env.horizon
    N = env.n_copies
    ds, ls = [], []
    for _ in range(episodes):
        s = env.reset()
        for t in range(H):
            st = augment(s, env.agent_pair_idx) if augment else s
            a = agent.act_batch(st, greedy=True)
            s, r, done, info = env.step(a)
        ds.append(env.last_lat.cpu().numpy())
        ls.append(env.last_los.cpu().numpy())
    return np.mean(np.stack(ds)), np.mean(np.stack(ls))


def rollout_ospf(env):
    s = env.reset()
    N = env.n_copies
    zeros = torch.zeros(N, dtype=torch.long, device=s.device)
    for t in range(env.horizon):
        s, r, done, info = env.step(zeros)
    return env.last_lat.cpu().numpy(), env.last_los.cpu().numpy()


def warmup_norms(env, act_fn=None):
    env.train_norm = True
    N = env.n_copies
    for _ in range(2):
        s = env.reset()
        for t in range(env.horizon):
            a = act_fn(s, env) if act_fn else torch.zeros(N, dtype=torch.long, device=s.device)
            s, _, _, _ = env.step(a)
    env.train_norm = False
    return env.norms()


# ---------------- pretrain ----------------

def run_pretrain_gpu(job, cfg, run_dir, device, seeds, n_copies, family):
    logger = setup_logger(run_dir, "gpre-" + os.path.basename(run_dir))
    mw = MetricsWriter(run_dir)
    env = build_vec_env(job, cfg, "source", seed=min(seeds), n_copies=n_copies, device=device)
    g = cfg["gpu"]
    if family == "ppo":
        agent = PPOGpuAgent(env.state_dim, env.action_dim, cfg, device=device,
                            use_pbrs=False, freeze_fraction=0.0, seeds=seeds,
                            minibatch=g["minibatch"])
        episodes = cfg[job["mode"]]["pretrain_episodes"]
        t0 = time.time()
        logger.info("pretrain(gpu,ppo,S=%d) topo=%s eps=%d copies=%d" % (len(seeds), job["topo"], episodes, env.n_copies))
        for ep in range(episodes):
            ep_r = rollout_ppo(agent, env)
            al = cl = None
            mw.write({"episode": ep, "phase": "source", "reward": float(ep_r.mean()),
                      "elapsed_s": round(time.time() - t0, 2)})
            if ep % max(1, episodes // 5) == 0:
                logger.info("pretrain ep=%d reward=%.4f dmax=%.4g lmax=%.4g" % (ep, float(ep_r.mean()), env.d_max, env.lam_max))
        ckpt = os.path.join(run_dir, "ckpt.pt")
        agent.save(ckpt, norms=env.norms())
    else:
        single = DQNGpuAgent if family == "dqn" else None
        assert single is not None, family
        agent = single(env.state_dim, env.action_dim, cfg, device=device, seed=min(seeds))
        episodes = cfg[job["mode"]]["pretrain_episodes"]
        t0 = time.time()
        for ep in range(episodes):
            rollout_dqn(agent, env, train=True)
            mw.write({"episode": ep, "phase": "source", "reward": "",
                      "elapsed_s": round(time.time() - t0, 2)})
        ckpt = os.path.join(run_dir, "ckpt.pt")
        agent.save(ckpt, norms=env.norms())
    write_manifest(run_dir, job, cfg, "pretrain_done")
    mark_done(run_dir)
    logger.info("pretrain done norms=%s" % (env.norms(),))
    return ckpt, env.norms()


def run_pretrain_grlps_gpu(job, cfg, run_dir, device, seed, n_copies):
    from .agents.grlps import spectral_embedding
    logger = setup_logger(run_dir, "gpre-" + os.path.basename(run_dir))
    mw = MetricsWriter(run_dir)
    env = build_vec_env(job, cfg, "source", seed=seed, n_copies=n_copies, device=device)
    emb = spectral_embedding(env.ug, cfg["grlps"]["embedding_dim"])
    agent = GRLPSGpuAgent(env.state_dim, env.action_dim, cfg, device=device, seed=seed, emb_matrix=emb)
    aug = lambda s, pair: agent.augment(s, pair)
    episodes = cfg[job["mode"]]["pretrain_episodes"]
    t0 = time.time()
    for ep in range(episodes):
        rollout_dqn(agent, env, train=True, augment=aug)
        mw.write({"episode": ep, "phase": "source", "reward": "", "elapsed_s": round(time.time() - t0, 2)})
    ckpt = os.path.join(run_dir, "ckpt.pt")
    agent.save(ckpt, norms=env.norms())
    write_manifest(run_dir, job, cfg, "pretrain_done")
    mark_done(run_dir)
    return ckpt, env.norms()


# ---------------- adaptation ----------------

def run_adapt_single(job, cfg, run_dir, device, ckpt=None):
    logger = setup_logger(run_dir, "adapt-" + os.path.basename(run_dir))
    mw = MetricsWriter(run_dir)
    method = job["method"]
    g = cfg["gpu"]
    n_copies = max(16, g["n_vec_envs"] // 8)
    env = build_vec_env(job, cfg, "target", seed=job["seed"], n_copies=n_copies, device=device)
    flags = method_flags("ppo_vanilla" if method == "ppo_local" else method)
    eval_eps = cfg[job["mode"]]["eval_episodes"]

    if method == "ospf":
        warmup_norms(env)
        lat, los = rollout_ospf(env)
        result = {"zero_shot": {"delay": float(lat.mean()), "loss": float(los.mean())},
                  "final": {"delay": float(lat.mean()), "loss": float(los.mean()),
                            "reward": None},
                  "converge_episode": 1, "norms": list(env.norms())}
        _write_eval(run_dir, job, cfg, result)
        return result

    if method in PPO_METHODS:
        vf = ppo_variant_flags("ppo_vanilla" if method == "ppo_local" else method)
        agent = PPOGpuAgent(env.state_dim, env.action_dim, cfg, device=device,
                            use_pbrs=vf["use_pbrs"], freeze_fraction=vf["freeze"],
                            seeds=[job["seed"]], minibatch=g["minibatch"])
    elif method == "dqn":
        agent = DQNGpuAgent(env.state_dim, env.action_dim, cfg, device=device, seed=job["seed"])
    elif method == "grlps":
        from .agents.grlps import spectral_embedding
        emb = spectral_embedding(env.ug, cfg["grlps"]["embedding_dim"])
        agent = GRLPSGpuAgent(env.state_dim, env.action_dim, cfg, device=device,
                              seed=job["seed"], emb_matrix=emb)
    else:
        raise ValueError(method)

    aug = agent.augment if method == "grlps" else None
    norms = None
    if flags["load_ckpt"] and ckpt and os.path.exists(ckpt):
        norms = agent.load(ckpt)
    if norms:
        env.d_max, env.lam_max = norms
        env.train_norm = False
    else:
        act = (lambda s, e: agent.act_batch(aug(s, e.agent_pair_idx) if aug else s)) \
            if method in ("dqn", "grlps") else (lambda s, e: agent.act(s)[0])
        warmup_norms(env, act_fn=act if method != "ospf" else None)
        norms = env.norms()

    episodes = cfg[job["mode"]]["adapt_episodes"]

    def do_eval():
        if method in PPO_METHODS:
            d, l = eval_ppo(agent, env, max(1, eval_eps // 2))
            return float(d[0]), float(l[0])
        d, l = eval_dqn(agent, env, max(1, eval_eps // 2), augment=aug)
        return float(d), float(l)

    zs_d, zs_l = do_eval()
    logger.info("zero-shot delay=%.4f loss=%.4f" % (zs_d, zs_l))
    t0 = time.time()
    rewards = []
    for ep in range(episodes):
        if method in PPO_METHODS:
            ep_r = rollout_ppo(agent, env)
            r_mean, d_mean, l_mean = float(ep_r.mean()), float(env.last_lat.mean()), float(env.last_los.mean())
        else:
            ep_r = rollout_dqn(agent, env, train=True, augment=aug)
            r_mean, d_mean, l_mean = float(ep_r.mean()), float(env.last_lat.mean()), float(env.last_los.mean())
        rewards.append(r_mean)
        row = {"episode": ep, "phase": "target", "reward": r_mean, "delay": d_mean,
               "loss": l_mean, "elapsed_s": round(time.time() - t0, 2)}
        if ep % max(1, episodes // 6) == 0:
            ed, el = do_eval()
            row["eval_delay"], row["eval_loss"] = ed, el
            logger.info("adapt ep=%d reward=%.4f eval_d=%.4f eval_l=%.4f" % (ep, r_mean, ed, el))
        mw.write(row)

    if method in PPO_METHODS:
        d, l = eval_ppo(agent, env, eval_eps)
        fd, fl = float(d[0]), float(l[0])
    else:
        fd, fl = eval_dqn(agent, env, eval_eps, augment=aug)
        fd, fl = float(fd), float(fl)
    result = {"zero_shot": {"delay": zs_d, "loss": zs_l},
              "final": {"delay": fd, "loss": fl, "reward": float(np.mean(rewards)) if rewards else None},
              "converge_episode": convergence_episode(rewards) if rewards else 1,
              "norms": list(env.norms())}
    if job.get("eval_rate_sweep"):
        sweep = {}
        for r in cfg["traffic"]["varying_rates"]:
            env.traffic_mode = "constant"
            env.rate = float(r)
            if method in PPO_METHODS:
                d, l = eval_ppo(agent, env, max(1, eval_eps // 2))
                sweep[r] = {"delay": float(d[0]), "loss": float(l[0])}
            else:
                d, l = eval_dqn(agent, env, max(1, eval_eps // 2), augment=aug)
                sweep[r] = {"delay": float(d), "loss": float(l)}
        result["rate_sweep"] = sweep
        env.traffic_mode = job.get("traffic_mode", "constant")
    _write_eval(run_dir, job, cfg, result)
    if hasattr(agent, "save"):
        try:
            agent.save(os.path.join(run_dir, "ckpt_adapted.pt"), norms=list(env.norms()))
        except Exception:
            pass
    return result


def run_adapt_ensemble(jobs, cfg, device, pretrain_ckpt, gpu_cfg):
    """jobs: list of per-seed job dicts sharing group_key. Trains S seeds stacked."""
    from .logging_utils import make_run_id
    method = jobs[0]["method"]
    vf = ppo_variant_flags("ppo_vanilla" if method == "ppo_local" else method)
    seeds = [j["seed"] for j in jobs]
    S = len(seeds)
    R = max(8, gpu_cfg["n_vec_envs"] // S)
    n_copies = S * R
    run_dirs = []
    for j in jobs:
        rd = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "logs", j["mode"], make_run_id(j))
        os.makedirs(rd, exist_ok=True)
        run_dirs.append(rd)
    loggers = [setup_logger(rd, "adapt-" + os.path.basename(rd)) for rd in run_dirs]
    mws = [MetricsWriter(rd) for rd in run_dirs]

    job = jobs[0]
    env = build_vec_env(job, cfg, "target", seed=min(seeds), n_copies=n_copies, device=device)
    eval_eps = cfg[job["mode"]]["eval_episodes"]
    agent = PPOGpuAgent(env.state_dim, env.action_dim, cfg, device=device,
                        use_pbrs=vf["use_pbrs"], freeze_fraction=vf["freeze"],
                        seeds=seeds, minibatch=gpu_cfg["minibatch"])
    norms = None
    flags = method_flags("ppo_vanilla" if method == "ppo_local" else method)
    if flags["load_ckpt"] and pretrain_ckpt and os.path.exists(pretrain_ckpt):
        norms = agent.load(pretrain_ckpt)
    if norms:
        env.d_max, env.lam_max = norms
        env.train_norm = False
    else:
        R0 = env.n_copies // agent.S
        warmup_norms(env, act_fn=lambda s, e: agent.act_ensemble(s.view(agent.S, R0, -1))[0].reshape(-1))
        norms = env.norms()

    def eval_all(episodes):
        d, l = eval_ppo(agent, env, episodes)
        return d, l

    zs_d, zs_l = eval_all(max(1, eval_eps // 2))
    zs0_d, zs0_l = zs_d.copy(), zs_l.copy()
    for i, lg in enumerate(loggers):
        lg.info("ensemble adapt S=%d R=%d method=%s zero-shot d=%.4f l=%.4f" % (S, R, method, zs_d[i], zs_l[i]))
    episodes = cfg[job["mode"]]["adapt_episodes"]
    t0 = time.time()
    rewards_per_seed = [[] for _ in range(S)]
    for ep in range(episodes):
        ep_r = rollout_ppo(agent, env)
        r_per = ep_r.view(S, R).mean(1).cpu().numpy()
        d_per = env.last_lat.view(S, R).mean(1).cpu().numpy()
        l_per = env.last_los.view(S, R).mean(1).cpu().numpy()
        if ep % max(1, episodes // 6) == 0:
            zs_d, zs_l = eval_all(max(1, eval_eps // 2))
            loggers[0].info("ens ep=%d mean_r=%.4f eval_d=%.4f eval_l=%.4f" % (ep, r_per.mean(), zs_d.mean(), zs_l.mean()))
        for i in range(S):
            rewards_per_seed[i].append(float(r_per[i]))
            row = {"episode": ep, "phase": "target", "reward": float(r_per[i]),
                   "delay": float(d_per[i]), "loss": float(l_per[i]),
                   "elapsed_s": round(time.time() - t0, 2)}
            if ep % max(1, episodes // 6) == 0:
                row["eval_delay"], row["eval_loss"] = float(zs_d[i]), float(zs_l[i])
            mws[i].write(row)
    fd, fl = eval_all(eval_eps)
    for i, j in enumerate(jobs):
        result = {"zero_shot": {"delay": float(zs0_d[i]), "loss": float(zs0_l[i])},
                  "final": {"delay": float(fd[i]), "loss": float(fl[i]),
                            "reward": float(np.mean(rewards_per_seed[i]))},
                  "converge_episode": convergence_episode(rewards_per_seed[i]),
                  "norms": list(env.norms())}
        if j.get("eval_rate_sweep"):
            sweep = {}
            for r in cfg["traffic"]["varying_rates"]:
                env.traffic_mode = "constant"
                env.rate = float(r)
                d, l = eval_all(max(1, eval_eps // 2))
                sweep[r] = {"delay": float(d[i]), "loss": float(l[i])}
            result["rate_sweep"] = sweep
            env.traffic_mode = j.get("traffic_mode", "constant")
        _write_eval(run_dirs[i], j, cfg, result)
    # save per-seed adapted checkpoints (single-model format)
    try:
        agent.save(os.path.join(run_dirs[0], "ckpt_adapted.pt"), norms=list(env.norms()))
    except Exception:
        pass
    return seeds


def _write_eval(run_dir, job, cfg, result):
    with open(os.path.join(run_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_manifest(run_dir, job, cfg, "done")
    mark_done(run_dir)
