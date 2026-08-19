"""GPU DQN and GRL-PS agents with on-device ring replay buffers."""
import numpy as np
import torch
import torch.nn as nn


class MLPQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNGpuAgent:
    def __init__(self, state_dim, action_dim, cfg, device="cuda", seed=0):
        torch.manual_seed(seed)
        self.cfg = cfg
        c = cfg["dqn"]
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.q = MLPQ(state_dim, action_dim, hidden=c["hidden_dim"]).to(self.device)
        self.target = MLPQ(state_dim, action_dim, hidden=c["hidden_dim"]).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=c["lr"])
        self.buf_size = c["buffer_size"]
        self.batch = c["batch_size"]
        self.sync_every = c["target_sync_steps"]
        self.eps_start, self.eps_end, self.eps_decay = c["eps_start"], c["eps_end"], c["eps_decay_steps"]
        self.gamma = cfg["paper_defaults"]["gamma"]
        self.state_dim = state_dim
        cap = self.buf_size
        self._s = torch.zeros(cap, state_dim, device=self.device)
        self._ns = torch.zeros(cap, state_dim, device=self.device)
        self._a = torch.zeros(cap, dtype=torch.long, device=self.device)
        self._r = torch.zeros(cap, device=self.device)
        self._d = torch.zeros(cap, device=self.device)
        self._ptr, self._full = 0, False
        self.steps = 0
        self.gen = torch.Generator(device=self.device.type)
        self.gen.manual_seed(int(seed) * 31337 + 5)

    def _eps(self):
        frac = min(1.0, self.steps / self.eps_decay)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    @torch.no_grad()
    def act_batch(self, states, greedy=False):
        """states [N, D] -> actions [N] (epsilon-greedy, shared epsilon)."""
        n = states.shape[0]
        qv = torch.argmax(self.q(states), dim=1)
        if not greedy:
            rand_mask = torch.rand(n, generator=self.gen, device=self.device) < self._eps()
            rand_a = torch.randint(self.action_dim, (n,), generator=self.gen, device=self.device)
            qv = torch.where(rand_mask, rand_a, qv)
        return qv

    def observe_batch(self, s, a, r, ns, d):
        n = s.shape[0]
        idx = torch.arange(self._ptr, self._ptr + n, device=self.device) % self.buf_size
        self._s[idx] = s
        self._ns[idx] = ns
        self._a[idx] = a
        self._r[idx] = r
        self._d[idx] = d
        self._ptr = int((self._ptr + n) % self.buf_size)
        self._full = self._full or self._ptr < n
        self.steps += n
        stored = self.buf_size if self._full else self._ptr
        if stored >= self.batch and self.steps % (self.batch * 4) < n:
            self._update()
        if self.steps % self.sync_every < n:
            self.target.load_state_dict(self.q.state_dict())

    def _update(self):
        stored = self.buf_size if self._full else self._ptr
        idx = torch.randint(stored, (self.batch,), generator=self.gen, device=self.device)
        s, ns = self._s[idx], self._ns[idx]
        a, r, d = self._a[idx], self._r[idx], self._d[idx]
        qv = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            y = r + self.gamma * (1 - d) * self.target(ns).max(1).values
        loss = torch.nn.functional.smooth_l1_loss(qv, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def save(self, path, norms=None):
        torch.save({"model": self.q.state_dict(), "norms": norms}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        sd = ckpt.get("model", ckpt)
        self.q.load_state_dict(sd)
        self.target.load_state_dict(sd)
        return ckpt.get("norms")


class GRLPSGpuAgent(DQNGpuAgent):
    """Simplified GRL-PS: spectral graph embedding prepended to the state."""

    def __init__(self, base_state_dim, action_dim, cfg, device="cuda", seed=0, emb_matrix=None):
        from .grlps import spectral_embedding  # reuse CPU-side embedding computation
        self.emb = torch.as_tensor(emb_matrix, dtype=torch.float32, device=device) if emb_matrix is not None else None
        self.emb_dim = cfg["grlps"]["embedding_dim"] if self.emb is not None else 0
        super().__init__(base_state_dim + 2 * self.emb_dim, action_dim, cfg, device, seed)

    def augment(self, states, pair_idx):
        """states [N, D], pair_idx [N, 2] node indices -> [N, D + 2E]."""
        if self.emb is None:
            return states
        e1 = self.emb[pair_idx[:, 0]]
        e2 = self.emb[pair_idx[:, 1]]
        return torch.cat([states, e1, e2], dim=1)
