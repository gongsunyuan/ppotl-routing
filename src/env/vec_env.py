"""GPU-vectorized network routing environment.

N environment copies live as batched tensors on one device. Queueing metrics use the
analytical M/M/1/K steady state (paper Eqs. 4-9) evaluated in float64 on GPU.

Design notes (differences vs the CPU env, by construction):
- All vectorized copies within one run share the same topology realization (link-failure
  pattern); traffic/flow randomness is per copy. In the CPU backend each run has a single
  copy anyway, so seed-level diversity is preserved where it matters (across runs).
- Under link failures the k-candidate table is approximated: original candidates that
  traverse a failed link are replaced by the current shortest path (per-source Dijkstra).
  Without failures the table is exact Yen k-shortest paths, cached per (topo, k).
"""
import os
from functools import lru_cache

import networkx as nx
import numpy as np
import torch

from .network_env import load_topology, k_shortest_paths


def mm1k_metrics_torch(arrivals, services, capacity):
    a = torch.as_tensor(arrivals, dtype=torch.float64)
    mu = torch.as_tensor(services, dtype=torch.float64)
    if mu.dim() < a.dim():
        mu = mu.expand_as(a) if a.shape[-1] == mu.numel() else mu.reshape(1, -1).expand_as(a)
    K = int(capacity)
    eps = 1e-9
    rho = a / mu.clamp(min=1e-12)
    P = torch.ones_like(rho)
    G = torch.zeros_like(rho)
    d = torch.zeros_like(rho)
    m_eq = (rho - 1.0).abs() < eps
    m_lo = (rho < 1.0 - eps) & (rho > eps)
    m_hi = rho > 1.0 + eps

    def branch_lo(r):
        rK = torch.exp(K * torch.log(r))
        rK1 = torch.exp((K + 1) * torch.log(r))
        return (1.0 - rK) / (1.0 - rK1), r / (1.0 - r) - (K + 1) * rK1 / (1.0 - rK1)

    def branch_hi(r):
        rmK = torch.exp(-K * torch.log(r))
        rmK1 = torch.exp(-(K + 1) * torch.log(r))
        return (1.0 / r) * (1.0 - rmK) / (1.0 - rmK1), -r / (r - 1.0) + (K + 1) / (1.0 - rmK1)

    P = torch.where(m_lo, torch.zeros_like(P), P)
    Pl, Gl = branch_lo(torch.where(m_lo, rho, torch.full_like(rho, 0.5)))
    P = torch.where(m_lo, Pl, P)
    G = torch.where(m_lo, Gl, G)
    Ph, Gh = branch_hi(torch.where(m_hi, rho, torch.full_like(rho, 2.0)))
    P = torch.where(m_hi, Ph, P)
    G = torch.where(m_hi, Gh, G)
    P = torch.where(m_eq, torch.full_like(P, K / (K + 1.0)), P)
    G = torch.where(m_eq, torch.full_like(G, K / 2.0), G)

    admitted = P * a
    d = torch.where(admitted > eps, G / admitted.clamp(min=1e-300), torch.zeros_like(d))
    loss = 1.0 - P
    return {"rho": rho, "P": P, "loss": loss, "delay": d, "queue": G}


@lru_cache(maxsize=32)
def _exact_table(topo_name, k):
    _, dg = load_topology(topo_name)
    nodes = list(dg.nodes())
    n = len(nodes)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and nx.has_path(dg, nodes[i], nodes[j])]
    cand_idx = np.full((len(pairs), k, n), -1, dtype=np.int64)
    for pi, (i, j) in enumerate(pairs):
        paths = k_shortest_paths(dg, nodes[i], nodes[j], k)
        for ci, p in enumerate(paths):
            for li, node in enumerate(p):
                cand_idx[pi, ci, li] = nodes.index(node)
    hops = (cand_idx >= 0).sum(axis=2).astype(np.float64)
    return pairs, cand_idx, hops


def _approx_table(topo_name, k, failed_edges):
    """Candidates under link failures: keep valid original candidates, fill with shortest path."""
    _, dg = load_topology(topo_name)
    nodes = list(dg.nodes())
    n = len(nodes)
    active = dg.copy()
    active.remove_edges_from([e for e in failed_edges if e in active.edges()])
    pairs, cand_idx, hops = _exact_table(topo_name, k)
    failed = set(failed_edges)
    name_to_idx = {name: i for i, name in enumerate(nodes)}
    shortest = {}
    for src in nodes:
        _, dist = nx.single_source_dijkstra(active, src)
        for dst in nodes:
            if dst == src or not nx.has_path(active, src, dst):
                continue
            sp = nx.shortest_path(active, src, dst)
            seq = [name_to_idx[v] for v in sp]
            for ci in range(k):
                shortest[(name_to_idx[src], name_to_idx[dst], ci)] = seq
    out = np.full_like(cand_idx, -1)
    for pi, (i, j) in enumerate(pairs):
        si, sj = nodes[i], nodes[j]
        if not nx.has_path(active, si, sj):
            for ci in range(k):
                seq = shortest.get((i, j, 0))
                if seq:
                    out[pi, ci, :len(seq)] = seq
            continue
        filled = 0
        for ci in range(k):
            seq = [v for v in cand_idx[pi, ci] if v >= 0]
            names = [nodes[int(v)] for v in seq]
            ok = all((names[t], names[t + 1]) not in failed for t in range(len(names) - 1))
            if ok and len(seq) > 0:
                out[pi, filled, :len(seq)] = seq
                filled += 1
        sp_seq = shortest.get((i, j, 0), [])
        while filled < k and sp_seq:
            out[pi, filled, :len(sp_seq)] = sp_seq
            filled += 1
    hops = (out >= 0).sum(axis=2).astype(np.float64)
    return pairs, out, hops


class VecNetEnv:
    """Batched routing environment: N copies on one torch device."""

    def __init__(self, topo_name, cfg, seed=0, traffic_mode="constant", rate=750,
                 fail_ratio=0.0, dynamic_failures=False, train_norm=True,
                 fixed_norm=None, k_paths=None, alpha1=None, alpha2=None,
                 n_copies=256, device="cuda"):
        self.topo_name = topo_name
        self.ug, self.dg = load_topology(topo_name)
        self.cfg = cfg
        self.k = k_paths or cfg["paper_defaults"]["k_paths"]
        self.alpha1 = cfg["paper_defaults"]["alpha1"] if alpha1 is None else alpha1
        self.alpha2 = cfg["paper_defaults"]["alpha2"] if alpha2 is None else alpha2
        self.K_buf = cfg["network"]["buffer_capacity"]
        self.M = cfg["network"]["n_background_flows"]
        self.traffic_mode = traffic_mode
        self.rate = float(rate)
        self.varying_rates = list(cfg["traffic"]["varying_rates"])
        self.fail_ratio = fail_ratio
        self.dynamic_failures = dynamic_failures
        self.train_norm = train_norm
        self.horizon = int(cfg.get("horizon", 200))
        self.device = torch.device(device)
        self.n_copies = int(n_copies)

        rng = np.random.default_rng(seed)
        self.nodes = list(self.ug.nodes())
        self.n_nodes = len(self.nodes)
        self.node_idx = {nm: i for i, nm in enumerate(self.nodes)}
        services = rng.choice(cfg["network"]["service_rates"], size=self.n_nodes).astype(np.float64)
        self.services = torch.as_tensor(services, dtype=torch.float64, device=self.device)

        if fixed_norm is not None:
            self.d_max, self.lam_max = float(fixed_norm[0]), float(fixed_norm[1])
        else:
            self.d_max, self.lam_max = 1e-8, 1e-8

        self._failed_edges = []
        if fail_ratio > 0:
            edges = list(self.dg.edges())
            n_fail = int(round(len(edges) * fail_ratio))
            rng.shuffle(edges)
            tmp = self.dg.copy()
            for e in edges[:n_fail]:
                tmp.remove_edge(*e)
                if not nx.is_strongly_connected(tmp):
                    tmp.add_edge(*e)
                else:
                    self._failed_edges.append(e)
        self._refresh_tables(approx=bool(self._failed_edges))
        self._fail_rng = np.random.default_rng(seed * 7 + 13)

        self.gen = torch.Generator(device=self.device.type)
        self.gen.manual_seed(int(seed) * 100003 + 17)
        self.t = 0
        self.state_dim = 2 + 1 + self.k + 2
        self.action_dim = self.k
        self._last_delay = torch.zeros(self.n_copies, self.n_nodes, dtype=torch.float64, device=self.device)
        self._last_loss = torch.zeros(self.n_copies, self.n_nodes, dtype=torch.float64, device=self.device)
        self._bg_pairs = torch.zeros(self.n_copies, self.M, dtype=torch.long, device=self.device)
        self.agent_pair_idx = torch.zeros(self.n_copies, 2, dtype=torch.long, device=self.device)
        self._sample_flows()
        self._eval_bg_only()
        self._agent_pi = self._sample_agent_flow()
        self.current_state = self._build_state()

    # ---------------- tables ----------------

    def _refresh_tables(self, approx=False):
        if approx:
            pairs, cand, hops = _approx_table(self.topo_name, self.k, tuple(sorted(self._failed_edges)))
        else:
            pairs, cand, hops = _exact_table(self.topo_name, self.k)
        self._pairs = pairs
        self._pair_pos = {p: i for i, p in enumerate(pairs)}
        P = len(pairs)
        self.n_pairs = P
        self._pair_src = torch.as_tensor([p[0] for p in pairs], dtype=torch.long, device=self.device)
        self._pair_dst = torch.as_tensor([p[1] for p in pairs], dtype=torch.long, device=self.device)
        self._cand_nodes = torch.as_tensor(cand, dtype=torch.long, device=self.device)
        mask = (self._cand_nodes >= 0)
        self._cand_mask = mask
        self._cand_nodes_safe = self._cand_nodes.clamp(min=0)
        self._cand_hops = torch.as_tensor(hops, dtype=torch.float64, device=self.device)
        self._cand_hops_norm = self._cand_hops / self._cand_hops.max(dim=1, keepdim=True).values.clamp(min=1)
        self._cand_mask_f = mask.to(torch.float64)

    def _random_failure(self):
        tmp = self.dg.copy()
        tmp.remove_edges_from(self._failed_edges)
        edges = list(tmp.edges())
        if not edges:
            return False
        e = edges[int(self._fail_rng.integers(len(edges)))]
        tmp.remove_edge(*e)
        if not nx.is_strongly_connected(tmp):
            return False
        self._failed_edges.append(e)
        self._refresh_tables(approx=True)
        return True

    # ---------------- flows ----------------

    def _current_rate(self):
        if self.traffic_mode == "varying":
            return float(self.varying_rates[self.t % len(self.varying_rates)])
        return self.rate

    def _rand_pair_indices(self, shape):
        return torch.randint(self.n_pairs, shape, generator=self.gen, device=self.device)

    def _sample_flows(self):
        self._bg_pairs = self._rand_pair_indices((self.n_copies, self.M))

    def _sample_agent_flow(self):
        pi = self._rand_pair_indices((self.n_copies,))
        self.agent_pair_idx = torch.stack([self._pair_src[pi], self._pair_dst[pi]], dim=1)
        r = float(self._current_rate())
        self._agent_rate = torch.poisson(
            torch.full((self.n_copies,), r, dtype=torch.float64, device=self.device), generator=self.gen)
        return pi
    # ---------------- evaluation ----------------

    def _aggregate(self, bg_pairs, agent_pair, agent_action, bg_rates, agent_rates):
        N = self.n_copies
        idx_bg = self._cand_nodes_safe[bg_pairs, 0]              # [N, M, L]
        mask_bg = self._cand_mask_f[bg_pairs, 0]                 # [N, M, L]
        rates_bg = bg_rates.unsqueeze(-1) * mask_bg              # [N, M, L]
        idx_ag = self._cand_nodes_safe[agent_pair, agent_action]  # [N, L]
        mask_ag = self._cand_mask_f[agent_pair, agent_action]
        rates_ag = agent_rates.unsqueeze(-1) * mask_ag

        arrivals = torch.zeros(N, self.n_nodes, dtype=torch.float64, device=self.device)
        flat_idx = torch.cat([idx_bg.reshape(N, -1), idx_ag], dim=1)
        flat_val = torch.cat([rates_bg.reshape(N, -1), rates_ag], dim=1)
        arrivals.scatter_add_(1, flat_idx, flat_val)

        met = mm1k_metrics_torch(arrivals, self.services, self.K_buf)
        self._last_delay = met["delay"]
        self._last_loss = met["loss"]
        delay = met["delay"]
        Pn = met["P"]

        gd = delay.unsqueeze(1).expand(N, self.M, self.n_nodes)
        d_bg = torch.gather(gd, 2, idx_bg) * mask_bg
        sum_d_bg = d_bg.sum(-1)                            # [N, M]
        p_bg = torch.gather(Pn.unsqueeze(1).expand(N, self.M, self.n_nodes), 2, idx_bg)
        surv_bg = torch.where(mask_bg > 0, p_bg, torch.ones_like(p_bg)).prod(-1)

        gd_ag = delay.unsqueeze(1).expand(N, 1, self.n_nodes).squeeze(1)
        d_ag = torch.gather(gd_ag, 1, idx_ag) * mask_ag
        sum_d_ag = d_ag.sum(-1)                            # [N]
        p_ag = torch.gather(Pn, 1, idx_ag)
        surv_ag = torch.where(mask_ag > 0, p_ag, torch.ones_like(p_ag)).prod(-1)

        total = (bg_rates.sum(1) + agent_rates).clamp(min=1e-12)
        w_bg = bg_rates / total.unsqueeze(1)
        w_ag = agent_rates / total
        lat = (w_bg * (sum_d_bg / (self.n_nodes * self.d_max))).sum(1) + w_ag * (sum_d_ag / (self.n_nodes * self.d_max))
        los = (w_bg * ((1.0 - surv_bg) / self.lam_max)).sum(1) + w_ag * ((1.0 - surv_ag) / self.lam_max)
        return met, lat, los

    def _eval_bg_only(self):
        N = self.n_copies
        r = float(self._current_rate())
        bg_rates = torch.poisson(
            torch.full((N, self.M), r, dtype=torch.float64, device=self.device), generator=self.gen)
        agent_pair = self._rand_pair_indices((N,))
        zero = torch.zeros(N, dtype=torch.float64, device=self.device)
        met, _, _ = self._aggregate(self._bg_pairs, agent_pair, torch.zeros(N, dtype=torch.long, device=self.device), bg_rates, zero)
        if self.train_norm:
            self.d_max = max(self.d_max, float(met["delay"].max()))
            self.lam_max = max(self.lam_max, float(met["loss"].max()))

    def _build_state(self):
        N = self.n_copies
        s = torch.zeros(N, self.state_dim, dtype=torch.float32, device=self.device)
        pi = self._agent_pi
        s[:, 0] = self._pair_src[pi].to(torch.float32) / self.n_nodes
        s[:, 1] = self._pair_dst[pi].to(torch.float32) / self.n_nodes
        r = float(self._current_rate())
        s[:, 2] = self._agent_rate.to(torch.float32) / float(r * (self.M + 1))
        s[:, 3:3 + self.k] = self._cand_hops_norm[pi].to(torch.float32)
        src = self._pair_src[pi]
        s[:, 3 + self.k] = (self._last_delay[torch.arange(N, device=self.device), src] / self.d_max).to(torch.float32)
        s[:, 4 + self.k] = (self._last_loss[torch.arange(N, device=self.device), src] / self.lam_max).to(torch.float32)
        return s

    # ---------------- API ----------------

    def reset(self):
        self.t = 0
        self._sample_flows()
        self._eval_bg_only()
        self._agent_pi = self._sample_agent_flow()
        self.current_state = self._build_state()
        return self.current_state

    def step(self, actions):
        N = self.n_copies
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).reshape(N)
        r = float(self._current_rate())
        bg_rates = torch.poisson(
            torch.full((N, self.M), r, dtype=torch.float64, device=self.device), generator=self.gen)
        met, lat, los = self._aggregate(self._bg_pairs, self._agent_pi, actions, bg_rates, self._agent_rate)
        if self.train_norm:
            self.d_max = max(self.d_max, float(met["delay"].max()))
            self.lam_max = max(self.lam_max, float(met["loss"].max()))
        reward = -(self.alpha1 * lat + self.alpha2 * los).to(torch.float32)
        self.last_lat = lat.detach()
        self.last_los = los.detach()
        self.last_reward = reward.detach()
        self.t += 1
        if self.dynamic_failures and self.t < self.horizon and self.horizon >= 4 \
                and self.t % max(1, self.horizon // 4) == 0:
            self._random_failure()
        self._sample_flows()
        self._agent_pi = self._sample_agent_flow()
        self.current_state = self._build_state()
        done = self.t >= self.horizon
        info = {"delay": float(lat.mean()), "loss": float(los.mean())}
        return self.current_state, reward, done, info

    def norms(self):
        return (self.d_max, self.lam_max)
