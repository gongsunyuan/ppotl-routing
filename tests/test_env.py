import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
import yaml

from src.env.network_env import NetworkEnv, mm1k_metrics, load_topology, potential
from src.agents.ppo import PPOAgent
from src.agents.nets import ActorCritic


def load_cfg():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "default.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["horizon"] = 10
    return cfg


def brute_mm1k(rho, K):
    if rho == 0:
        return 1.0, 0.0
    ps = np.array([rho ** n for n in range(K + 1)], dtype=np.float64)
    ps = ps / ps.sum()
    mean_q = float((np.arange(K + 1) * ps).sum())
    p_loss = float(ps[K])
    return 1.0 - p_loss, mean_q


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.5, 5.0, 50.0])
def test_mm1k_against_brute_force(rho):
    K = 100
    mu = 1000.0
    a = rho * mu
    m = mm1k_metrics([a], [mu], K)
    P_ref, G_ref = brute_mm1k(rho, K)
    assert m["P"][0] == pytest.approx(P_ref, abs=1e-6)
    assert m["queue"][0] == pytest.approx(G_ref, abs=1e-4)


def test_mm1k_zero_arrival():
    m = mm1k_metrics([0.0], [1000.0], 100)
    assert m["P"][0] == 1.0
    assert m["delay"][0] == 0.0
    assert m["loss"][0] == 0.0


def test_topology_counts():
    expect = {"abilene": (11, 28), "cernet": (41, 116), "claranet": (15, 36), "nsfcnet": (9, 20)}
    for name, (n, m) in expect.items():
        ug, dg = load_topology(name)
        assert ug.number_of_nodes() == n
        assert dg.number_of_edges() == m


def test_env_state_action_reward():
    cfg = load_cfg()
    env = NetworkEnv("nsfcnet", cfg, seed=0, traffic_mode="constant", rate=500)
    s = env.reset()
    assert s.shape[0] == env.state_dim
    assert np.all(np.isfinite(s))
    assert 0 <= s[0] <= 1 and 0 <= s[1] <= 1
    s2, r, done, info = env.step(env.action_dim - 1)
    assert np.all(np.isfinite(s2))
    assert np.isfinite(r)
    assert info["delay"] >= 0 and info["loss"] >= 0


def test_pbrs_telescoping():
    cfg = load_cfg()
    agent = PPOAgent(9, 4, cfg, seed=0, use_pbrs=True)
    rng = np.random.default_rng(0)
    T = 6
    states = rng.random((T + 1, 9))
    rewards = rng.random(T)
    shaped = np.array([agent.shape_reward(r, states[i], states[i + 1]) for i, r in enumerate(rewards)])
    g = 0.99
    phi = lambda x: agent.zeta1 * (1 - x[-2]) + agent.zeta2 * (1 - x[-1])
    total = sum((g ** i) * r for i, r in enumerate(rewards)) + (g ** T) * phi(states[-1]) - phi(states[0])
    acc = 0.0
    for i in range(T):
        acc += (g ** i) * shaped[i]
    assert acc == pytest.approx(total, abs=1e-10)


def test_frozen_backbone_zero_grad():
    model = ActorCritic(9, 4, hidden=32)
    model.set_frozen_fraction(1.0)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randn(16, 9)
    dist, v = model.act(x)
    loss = -dist.log_prob(torch.randint(0, 4, (16,))).mean() + v.mean() ** 2
    opt.zero_grad()
    loss.backward()
    for p in model.backbone.parameters():
        if not p.requires_grad:
            assert p.grad is None or torch.all(p.grad == 0)
    frozen, total = model.frozen_parameter_count()
    assert frozen > 0 and frozen < total


def test_partial_freeze():
    model = ActorCritic(9, 4, hidden=32, n_backbone_layers=2)
    model.set_frozen_fraction(0.5)
    l0 = model.backbone[0].weight
    l2 = model.backbone[2].weight
    assert not l0.requires_grad and l2.requires_grad


def test_failure_keeps_connectivity():
    cfg = load_cfg()
    env = NetworkEnv("abilene", cfg, seed=3, fail_ratio=0.3, dynamic_failures=True)
    import networkx as nx
    env.reset()
    for _ in range(20):
        env.step(0)
        assert nx.is_strongly_connected(env._active)
