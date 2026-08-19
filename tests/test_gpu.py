import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
import yaml

from src.env.network_env import mm1k_metrics as mm1k_np
from src.env.vec_env import mm1k_metrics_torch, VecNetEnv


def load_cfg():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "default.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["horizon"] = 10
    return cfg


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.5, 5.0, 50.0])
def test_mm1k_torch_matches_numpy(rho):
    K = 200
    mu = np.array([1000.0, 2000.0])
    a = np.array([rho * 1000.0, rho * 2000.0])
    mnp = mm1k_np(a, mu, K)
    mt = mm1k_metrics_torch(torch.tensor(a).unsqueeze(0),
                            torch.tensor(mu), K)
    for key in ("P", "loss", "delay", "queue"):
        np.testing.assert_allclose(mt[key][0].numpy(), mnp[key], rtol=1e-9, atol=1e-9)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_vec_env_step_shapes():
    cfg = load_cfg()
    env = VecNetEnv("nsfcnet", cfg, seed=0, rate=500, n_copies=8, device=_device())
    s = env.reset()
    assert s.shape == (8, env.state_dim)
    assert torch.isfinite(s).all()
    a = torch.zeros(8, dtype=torch.long, device=s.device)
    s2, r, done, info = env.step(a)
    assert s2.shape == s.shape and r.shape == (8,)
    assert torch.isfinite(r).all()
    assert info["delay"] >= 0 and info["loss"] >= 0


def test_vec_env_deterministic_same_seed():
    cfg = load_cfg()
    dev = _device()
    def run():
        env = VecNetEnv("nsfcnet", cfg, seed=3, rate=500, n_copies=8, device=dev)
        torch.manual_seed(0)
        rs = []
        s = env.reset()
        for _ in range(5):
            a = torch.randint(env.action_dim, (8,), device=s.device)
            s, r, done, info = env.step(a)
            rs.append(r.cpu())
        return torch.stack(rs)
    x, y = run(), run()
    assert torch.allclose(x, y)


def test_vec_env_matches_cpu_stats():
    """Same traffic intensity -> same steady-state metric distribution (loose tolerance)."""
    cfg = load_cfg()
    dev = _device()
    env = VecNetEnv("nsfcnet", cfg, seed=0, rate=750, n_copies=64, device=dev, train_norm=True)
    s = env.reset()
    for _ in range(10):
        s, r, done, info = env.step(torch.zeros(64, dtype=torch.long, device=s.device))
    assert 0 < info["delay"] < 1e3
    assert 0 <= info["loss"] <= env.n_nodes


def test_batched_ensemble_matches_single():
    """Stacked training must be seed-independent: slice s of an S-stack behaves like a
    single model initialized with the same seed after identical updates."""
    from src.agents.batched_nets import BatchedActorCritic
    torch.manual_seed(0)
    D, A, B = 11, 4, 32
    single = BatchedActorCritic(D, A, S=1, base_seed=7).cpu()
    multi = BatchedActorCritic(D, A, S=3, base_seed=7).cpu()
    x1 = torch.randn(B, D)
    x3 = torch.randn(3, B, D)
    x3[0] = x1  # seed 0 of the stack sees exactly the single model's batch
    with torch.no_grad():
        l1, v1 = single.forward_seed(x1, 0)
        l3, v3 = multi.forward_seed(x1, 0)
    assert torch.allclose(l1, l3, atol=1e-6)
    assert torch.allclose(v1, v3, atol=1e-6)

    single.zero_grad()
    multi.zero_grad()
    t1 = single(x1.unsqueeze(0))
    (t1[0].sum() + t1[1].sum()).backward()
    t3 = multi(x3)
    (t3[0].sum() + t3[1].sum()).backward()
    for k in dict(single.named_parameters()):
        g1 = dict(single.named_parameters())[k].grad
        g3 = dict(multi.named_parameters())[k].grad
        assert torch.allclose(g1, g3[0], atol=1e-6), k
        assert g3[1].abs().sum() > 0 or g3[2].abs().sum() > 0, "other seeds must still receive gradients"


def test_freeze_masks_gpu_ppo():
    from src.agents.ppo_gpu import PPOGpuAgent
    cfg = load_cfg()
    cfg["gpu"] = {"minibatch": 64}
    ag = PPOGpuAgent(11, 4, cfg, device=_device(), freeze_fraction=1.0, seeds=(0,))
    frozen, total = ag.model.frozen_parameter_count()
    assert frozen > 0 and frozen < total
    for name, m in ag.masks.items():
        if name.startswith(("b0", "b1")):
            assert m.abs().sum().item() == 0
        else:
            assert m.abs().sum().item() == m.numel()
