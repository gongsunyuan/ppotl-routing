import json
import os
import networkx as nx
import numpy as np

TOPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "topo")


def load_topology(name):
    path = os.path.join(TOPO_DIR, name + ".json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ug = nx.Graph()
    ug.add_nodes_from(data["nodes"])
    ug.add_edges_from(data["edges"])
    dg = ug.to_directed()
    return ug, dg


def k_shortest_paths(dg, src, dst, k):
    paths = []
    try:
        gen = nx.shortest_simple_paths(dg, src, dst)
        for p in gen:
            paths.append(p)
            if len(paths) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    while len(paths) < k:
        paths.append(paths[-1] if paths else [src, dst])
    return paths


def mm1k_metrics(arrivals, services, capacity):
    a = np.asarray(arrivals, dtype=np.float64)
    mu = np.asarray(services, dtype=np.float64)
    K = int(capacity)
    rho = np.zeros_like(a)
    mask_mu = mu > 0
    rho[mask_mu] = a[mask_mu] / mu[mask_mu]
    P = np.ones_like(rho)
    G = np.zeros_like(rho)
    d = np.zeros_like(rho)
    eps = 1e-9
    m_eq = np.abs(rho - 1.0) < eps
    m_lo = (rho < 1.0 - eps) & (rho > eps)
    m_hi = rho > 1.0 + eps
    if np.any(m_lo):
        r = rho[m_lo]
        rK = np.exp(K * np.log(r))
        rK1 = np.exp((K + 1) * np.log(r))
        P[m_lo] = (1.0 - rK) / (1.0 - rK1)
        G[m_lo] = r / (1.0 - r) - (K + 1) * rK1 / (1.0 - rK1)
    if np.any(m_hi):
        r = rho[m_hi]
        rmK = np.exp(-K * np.log(r))
        rmK1 = np.exp(-(K + 1) * np.log(r))
        P[m_hi] = (1.0 / r) * (1.0 - rmK) / (1.0 - rmK1)
        G[m_hi] = -r / (r - 1.0) + (K + 1) / (1.0 - rmK1)
    if np.any(m_eq):
        P[m_eq] = K / (K + 1.0)
        G[m_eq] = K / 2.0
    admitted = P * a
    m_adm = admitted > eps
    d[m_adm] = G[m_adm] / admitted[m_adm]
    loss = 1.0 - P
    return {"rho": rho, "P": P, "loss": loss, "delay": d, "queue": G}


def potential(state, zeta1, zeta2, idx_delay=-2, idx_loss=-1):
    return zeta1 * (1.0 - state[idx_delay]) + zeta2 * (1.0 - state[idx_loss])


class NetworkEnv:
    def __init__(self, topo_name, cfg, seed=0, traffic_mode="constant", rate=750,
                 fail_ratio=0.0, dynamic_failures=False, train_norm=True,
                 fixed_norm=None, k_paths=None, alpha1=None, alpha2=None):
        self.topo_name = topo_name
        self.ug, self.dg = load_topology(topo_name)
        self.cfg = cfg
        self.k = k_paths or cfg["paper_defaults"]["k_paths"]
        self.alpha1 = cfg["paper_defaults"]["alpha1"] if alpha1 is None else alpha1
        self.alpha2 = cfg["paper_defaults"]["alpha2"] if alpha2 is None else alpha2
        self.K_buf = cfg["network"]["buffer_capacity"]
        self.M = cfg["network"]["n_background_flows"]
        self.service_choices = cfg["network"]["service_rates"]
        self.traffic_mode = traffic_mode
        self.rate = float(rate)
        self.varying_rates = cfg["traffic"]["varying_rates"]
        self.fail_ratio = fail_ratio
        self.dynamic_failures = dynamic_failures
        self.train_norm = train_norm
        self.rng = np.random.default_rng(seed)
        self.nodes = list(self.ug.nodes())
        self.n_nodes = len(self.nodes)
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.services = self.rng.choice(self.service_choices, size=self.n_nodes).astype(np.float64)
        if fixed_norm is not None:
            self.d_max, self.lam_max = float(fixed_norm[0]), float(fixed_norm[1])
        else:
            self.d_max, self.lam_max = 1e-8, 1e-8
        self.horizon = int(cfg.get("horizon", 200))
        self._path_cache = {}
        self._active = self.dg.copy()
        self._apply_initial_failures()
        self.bg_flows = []
        self.bg_rates = None
        self.t = 0
        self.current_state = None
        self.state_dim = 2 + 1 + self.k + 2
        self.action_dim = self.k
        self._last_node_delay = np.zeros(self.n_nodes)
        self._last_node_loss = np.zeros(self.n_nodes)
        self._roll_flows()
        self._sample_agent_flow()
        self.current_state = self._build_state()

    def _apply_initial_failures(self):
        if self.fail_ratio <= 0:
            return
        edges = list(self._active.edges())
        n_fail = int(round(len(edges) * self.fail_ratio))
        self.rng.shuffle(edges)
        for e in edges[:n_fail]:
            self._active.remove_edge(*e)
            if not nx.is_strongly_connected(self._active):
                self._active.add_edge(*e)

    def _apply_random_failure(self):
        edges = list(self._active.edges())
        if not edges:
            return False
        e = edges[int(self.rng.integers(len(edges)))]
        self._active.remove_edge(*e)
        if not nx.is_strongly_connected(self._active):
            self._active.add_edge(*e)
            return False
        self._path_cache = {}
        return True

    def _roll_flows(self):
        pairs = []
        for _ in range(self.M):
            while True:
                s = self.nodes[int(self.rng.integers(self.n_nodes))]
                d = self.nodes[int(self.rng.integers(self.n_nodes))]
                if s != d and nx.has_path(self._active, s, d):
                    break
            pairs.append((s, d))
        self.bg_flows = pairs

    def _get_candidates(self, src, dst):
        key = (src, dst)
        if key not in self._path_cache:
            self._path_cache[key] = k_shortest_paths(self._active, src, dst, self.k)
        return self._path_cache[key]

    def _current_rate(self):
        if self.traffic_mode == "varying":
            return float(self.varying_rates[self.t % len(self.varying_rates)])
        return self.rate

    def _bg_paths(self):
        res = []
        for s, d in self.bg_flows:
            p = nx.shortest_path(self._active, s, d)
            res.append((s, d, p))
        return res

    def _build_state(self):
        src, dst = self.agent_flow
        cands = self._get_candidates(src, dst)
        hops = np.array([len(p) - 1 for p in cands], dtype=np.float64)
        norm = hops.max() if hops.max() > 0 else 1.0
        rate_bg = self._current_rate() * self.M
        node_v = self.node_idx[src]
        d_v = self._last_node_delay[node_v]
        l_v = self._last_node_loss[node_v]
        phi = self.agent_rate / (self.agent_rate + rate_bg) if (self.agent_rate + rate_bg) > 0 else 0.0
        s = np.zeros(self.state_dim, dtype=np.float64)
        s[0] = self.node_idx[src] / self.n_nodes
        s[1] = self.node_idx[dst] / self.n_nodes
        s[2] = phi
        s[3:3 + self.k] = hops / norm
        s[3 + self.k] = d_v
        s[4 + self.k] = l_v
        return s

    def reset(self):
        self.t = 0
        self._active = self.dg.copy()
        self._apply_initial_failures()
        self._path_cache = {}
        self._roll_flows()
        arrivals = self.rng.poisson(self._current_rate(), size=self.M).astype(np.float64)
        self.bg_rates = arrivals
        self._evaluate_network(self._bg_paths(), None)
        self._sample_agent_flow()
        self.current_state = self._build_state()
        return self.current_state

    def _sample_agent_flow(self):
        while True:
            s = self.nodes[int(self.rng.integers(self.n_nodes))]
            d = self.nodes[int(self.rng.integers(self.n_nodes))]
            if s != d and nx.has_path(self._active, s, d):
                break
        self.agent_flow = (s, d)
        self.agent_rate = float(self.rng.poisson(self._current_rate()))

    def _evaluate_network(self, bg_paths, agent_path):
        link_load = {}
        node_arrival = np.zeros(self.n_nodes)
        for i, (s, d, p) in enumerate(bg_paths):
            r = self.bg_rates[i] if self.bg_rates is not None else self._current_rate()
            for u, v in zip(p[:-1], p[1:]):
                link_load[(u, v)] = link_load.get((u, v), 0.0) + r
                node_arrival[self.node_idx[v]] += r
        if agent_path is not None:
            for u, v in zip(agent_path[:-1], agent_path[1:]):
                link_load[(u, v)] = link_load.get((u, v), 0.0) + self.agent_rate
                node_arrival[self.node_idx[v]] += self.agent_rate
        met = mm1k_metrics(node_arrival, self.services, self.K_buf)
        self._last_node_delay = met["delay"]
        self._last_node_loss = met["loss"]
        if self.train_norm:
            self.d_max = max(self.d_max, float(met["delay"].max()) if met["delay"].size else self.d_max)
            self.lam_max = max(self.lam_max, float(met["loss"].max()) if met["loss"].size else self.lam_max)
        flows = []
        total_rate = 0.0
        for i, (s, d, p) in enumerate(bg_paths):
            r = self.bg_rates[i] if self.bg_rates is not None else self._current_rate()
            dv = sum(met["delay"][self.node_idx[v]] for v in p[1:])
            sv = 1.0
            for v in p[1:]:
                sv *= met["P"][self.node_idx[v]]
            flows.append((r, dv, 1.0 - sv))
            total_rate += r
        if agent_path is not None:
            dv = sum(met["delay"][self.node_idx[v]] for v in agent_path[1:])
            sv = 1.0
            for v in agent_path[1:]:
                sv *= met["P"][self.node_idx[v]]
            flows.append((self.agent_rate, dv, 1.0 - sv))
            total_rate += self.agent_rate
        lat = 0.0
        los = 0.0
        if total_rate > 0:
            for r, dv, lv in flows:
                w = r / total_rate
                lat += w * (dv / (self.n_nodes * self.d_max))
                los += w * (lv / self.lam_max)
        info = {"delay": lat, "loss": los, "raw_delay": lat * self.n_nodes * self.d_max,
                "raw_loss": los * self.lam_max}
        return info

    def step(self, action):
        src, dst = self.agent_flow
        cands = self._get_candidates(src, dst)
        agent_path = cands[int(action) % len(cands)]
        self.bg_rates = self.rng.poisson(self._current_rate(), size=self.M).astype(np.float64)
        info = self._evaluate_network(self._bg_paths(), agent_path)
        reward = -(self.alpha1 * info["delay"] + self.alpha2 * info["loss"])
        self.t += 1
        if self.dynamic_failures and self.t % max(1, self.horizon // 4) == 0 and self.t < self.horizon:
            self._apply_random_failure()
        self._sample_agent_flow()
        self.current_state = self._build_state()
        done = self.t >= self.horizon
        return self.current_state, reward, done, info

    def ospf_action(self, state=None):
        src, dst = self.agent_flow
        cands = self._get_candidates(src, dst)
        return 0

    def norms(self):
        return (self.d_max, self.lam_max)
