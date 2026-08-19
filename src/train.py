import json
import os
import time
import numpy as np

from .env.network_env import NetworkEnv
from .agents.ppo import PPOAgent
from .agents.dqn import DQNAgent
from .agents.grlps import GRLPSAgent, spectral_embedding
from .baselines.ospf import OSPFPolicy
from .logging_utils import MetricsWriter, setup_logger, write_manifest, mark_done
from .evaluate import evaluate_policy, convergence_episode

PPO_METHODS = {"ppo_vanilla", "ppo_naive", "ppotl", "ppotl_freeze_only", "ppotl_pbrs_only", "ppo_local"}


def make_agent(method, env, cfg, seed, topo=None):
    if method == "ospf":
        return OSPFPolicy(cfg)
    if method in PPO_METHODS:
        use_pbrs = method in {"ppotl", "ppotl_pbrs_only"}
        freeze = 1.0 if method in {"ppotl", "ppotl_freeze_only"} else 0.0
        agent = PPOAgent(env.state_dim, env.action_dim, cfg, seed=seed,
                         use_pbrs=use_pbrs, freeze_fraction=freeze)
        return agent
    if method == "dqn":
        return DQNAgent(env.state_dim, env.action_dim, cfg, seed=seed)
    if method == "grlps":
        emb = spectral_embedding(env.ug, cfg["grlps"]["embedding_dim"]) if topo else None
        return GRLPSAgent(env.state_dim, env.action_dim, cfg, seed=seed, embedding_matrix=emb)
    raise ValueError(method)


def method_flags(method):
    return {
        "load_ckpt": method in {"ppo_naive", "ppotl", "ppotl_freeze_only", "ppotl_pbrs_only", "grlps"},
        "needs_pretrain": method in {"ppo_naive", "ppotl", "ppotl_freeze_only", "ppotl_pbrs_only", "grlps"},
        "adapts": method != "ospf",
    }


def build_env(job, cfg, phase, seed):
    mode_cfg = cfg[job["mode"]]
    horizon = mode_cfg["horizon"]
    local = dict(cfg)
    local["horizon"] = horizon
    topo = job.get("target_topo") or job["topo"]
    traffic_mode = job.get("traffic_mode", "constant")
    rate = job.get("rate", cfg["traffic"]["source_mean_rate"]) if phase == "target" else cfg["traffic"]["source_mean_rate"]
    if traffic_mode == "varying":
        rate = cfg["traffic"]["source_mean_rate"]
    env = NetworkEnv(
        topo, local, seed=seed,
        traffic_mode=traffic_mode,
        rate=rate,
        fail_ratio=job.get("fail_ratio", 0.0) if phase == "target" else 0.0,
        dynamic_failures=bool(job.get("dynamic_failures", False)) and phase == "target",
        train_norm=phase == "source",
        fixed_norm=job.get("fixed_norm") if phase == "target" else None,
        k_paths=job.get("k_paths"),
        alpha1=job.get("alpha1"),
    )
    return env


def run_pretrain(job, cfg, run_dir):
    logger = setup_logger(run_dir, "pretrain")
    mw = MetricsWriter(run_dir)
    env = build_env(job, cfg, "source", job["seed"])
    method = job["method"]
    if method in PPO_METHODS:
        agent = PPOAgent(env.state_dim, env.action_dim, cfg, seed=job["seed"],
                         use_pbrs=False, freeze_fraction=0.0)
        aug = None
    elif method == "grlps":
        emb = spectral_embedding(env.ug, cfg["grlps"]["embedding_dim"])
        agent = GRLPSAgent(env.state_dim, env.action_dim, cfg, seed=job["seed"], embedding_matrix=emb)
        aug = _grlps_aug(agent, env)
    else:
        raise ValueError("method %s does not support pretraining" % method)
    episodes = cfg[job["mode"]]["pretrain_episodes"]
    logger.info("pretrain method=%s topo=%s seed=%d episodes=%d" % (method, job["topo"], job["seed"], episodes))
    t0 = time.time()
    for ep in range(episodes):
        res = run_episode(agent, env, aug, train=True)
        mw.write({"episode": ep, "phase": "source", **{k: res.get(k) for k in ("reward", "delay", "loss", "actor_loss", "critic_loss")}, "elapsed_s": round(time.time() - t0, 2)})
        if ep % max(1, episodes // 5) == 0:
            logger.info("pretrain ep=%d reward=%.4f delay=%.4f loss=%.4f" % (ep, res["reward"], res["delay"], res["loss"]))
    ckpt = os.path.join(run_dir, "ckpt.pt")
    agent.save(ckpt, norms=env.norms())
    logger.info("pretrain done norms=%s" % (env.norms(),))
    write_manifest(run_dir, job, cfg, "pretrain_done")
    mark_done(run_dir)
    return ckpt, env.norms()


def _grlps_aug(agent, env):
    def aug(state, e):
        src, dst = e.agent_flow
        return agent.augment(state, e.node_idx[src], e.node_idx[dst])
    return aug


def run_episode(agent, env, aug, train=True):
    s = env.reset()
    states, actions, logps, rewards, nexts = [], [], [], [], []
    raw_rewards, delays, losses = [], [], []
    done = False
    while not done:
        st = aug(s, env) if aug else s
        a, logp = agent.select_action(st, greedy=not train)
        s2, r, done, info = env.step(a)
        if train and isinstance(agent, PPOAgent):
            r_shaped = agent.shape_reward(r, s, s2)
        else:
            r_shaped = r
        if train and isinstance(agent, PPOAgent):
            states.append(s); actions.append(a); logps.append(logp); rewards.append(r_shaped); nexts.append(s2)
        if train and isinstance(agent, (DQNAgent, GRLPSAgent)):
            st2 = aug(s2, env) if aug else s2
            agent.observe(st, a, r, st2, done)
        raw_rewards.append(r); delays.append(info["delay"]); losses.append(info["loss"])
        s = s2
    out = {"reward": float(np.mean(raw_rewards)), "delay": float(np.mean(delays)), "loss": float(np.mean(losses))}
    if train and isinstance(agent, PPOAgent) and states:
        al, cl = agent.update(states, actions, logps, rewards, nexts)
        out["actor_loss"], out["critic_loss"] = al, cl
    return out


def run_adapt(job, cfg, run_dir, ckpt=None):
    logger = setup_logger(run_dir, "adapt")
    mw = MetricsWriter(run_dir)
    method = job["method"]
    env = build_env(job, cfg, "target", job["seed"])
    flags = method_flags(method)
    agent = make_agent(method, env, cfg, job["seed"], topo=job.get("target_topo") or job["topo"])
    aug = None
    if method == "grlps":
        emb = spectral_embedding(env.ug, cfg["grlps"]["embedding_dim"])
        agent.emb_matrix = emb
        aug = _grlps_aug(agent, env)
    norms = None
    if flags["load_ckpt"] and ckpt and os.path.exists(ckpt):
        norms = agent.load(ckpt)
    if norms:
        env.d_max, env.lam_max = norms
        env.train_norm = False
    else:
        env.train_norm = True
        for _ in range(2):
            s = env.reset()
            done = False
            while not done:
                st = aug(s, env) if aug else s
                a, _ = agent.select_action(st, greedy=False)
                s, _, done, _ = env.step(a)
        env.train_norm = False
        norms = env.norms()
    episodes = cfg[job["mode"]]["adapt_episodes"] if flags["adapts"] else 1
    eval_eps = max(1, cfg[job["mode"]]["eval_episodes"] // 2)
    logger.info("adapt method=%s topo=%s seed=%d episodes=%d mode=%s rate=%s fail=%s" %
                (method, job.get("target_topo") or job["topo"], job["seed"], episodes,
                 job.get("traffic_mode", "constant"), job.get("rate"), job.get("fail_ratio", 0)))
    eval_agent = agent.dqn if method == "grlps" else agent
    zs = evaluate_policy(eval_agent, env, eval_eps, aug=aug) if method != "ospf" else evaluate_policy(agent, env, eval_eps)
    logger.info("zero-shot eval %s" % zs)
    t0 = time.time()
    rewards = []
    for ep in range(episodes):
        res = run_episode(agent, env, aug, train=flags["adapts"])
        rewards.append(res["reward"])
        row = {"episode": ep, "phase": "target", **{k: res.get(k) for k in ("reward", "delay", "loss", "actor_loss", "critic_loss")}, "elapsed_s": round(time.time() - t0, 2)}
        if ep % max(1, episodes // 6) == 0:
            ev = evaluate_policy(eval_agent, env, eval_eps, aug=aug)
            row["eval_delay"], row["eval_loss"] = ev["delay"], ev["loss"]
            logger.info("adapt ep=%d reward=%.4f eval_delay=%.4f eval_loss=%.4f" % (ep, res["reward"], ev["delay"], ev["loss"]))
        mw.write(row)
    final = evaluate_policy(eval_agent, env, cfg[job["mode"]]["eval_episodes"], aug=aug)
    result = {"zero_shot": zs, "final": final,
              "converge_episode": convergence_episode(rewards) if rewards else 0,
              "norms": [env.d_max, env.lam_max]}
    if job.get("eval_rate_sweep"):
        sweep = {}
        for r in cfg["traffic"]["varying_rates"]:
            env.traffic_mode = "constant"
            env.rate = float(r)
            ev = evaluate_policy(eval_agent, env, eval_eps, aug=aug)
            sweep[r] = {"delay": ev["delay"], "loss": ev["loss"], "delay_std": ev["delay_std"], "loss_std": ev["loss_std"]}
        result["rate_sweep"] = sweep
        env.traffic_mode = job.get("traffic_mode", "constant")
    with open(os.path.join(run_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("final eval %s" % final)
    if method in PPO_METHODS:
        ck = os.path.join(run_dir, "ckpt_adapted.pt")
        agent.save(ck, norms=[env.d_max, env.lam_max])
    write_manifest(run_dir, job, cfg, "done")
    mark_done(run_dir)
    return result
