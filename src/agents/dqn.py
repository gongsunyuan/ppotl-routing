import random
from collections import deque
import numpy as np
import torch
from .nets import QNet


class DQNAgent:
    needs_pretrain = True

    def __init__(self, state_dim, action_dim, cfg, device="cpu", seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        self.cfg = cfg
        self.action_dim = action_dim
        c = cfg["dqn"]
        self.q = QNet(state_dim, action_dim, hidden=c["hidden_dim"]).to(device)
        self.target = QNet(state_dim, action_dim, hidden=c["hidden_dim"]).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=c["lr"])
        self.buffer = deque(maxlen=c["buffer_size"])
        self.batch_size = c["batch_size"]
        self.target_sync = c["target_sync_steps"]
        self.eps_start, self.eps_end = c["eps_start"], c["eps_end"]
        self.eps_decay = c["eps_decay_steps"]
        self.gamma = cfg["paper_defaults"]["gamma"]
        self.steps = 0

    def _eps(self):
        frac = min(1.0, self.steps / self.eps_decay)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self._eps():
            return random.randrange(self.action_dim), 0.0
        with torch.no_grad():
            qv = self.q(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(qv, dim=1).item()), 0.0

    def observe(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, float(done)))
        self.steps += 1
        if len(self.buffer) >= self.batch_size and self.steps % 4 == 0:
            self._update()
        if self.steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())

    def _update(self):
        batch = random.sample(self.buffer, self.batch_size)
        S = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        A = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
        R = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        SN = torch.as_tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        D = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)
        q_vals = self.q(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nxt = self.target(SN).max(dim=1).values
            y = R + self.gamma * (1 - D) * nxt
        loss = torch.nn.functional.smooth_l1_loss(q_vals, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save(self, path, norms=None):
        torch.save({"model": self.q.state_dict(), "norms": norms}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.q.load_state_dict(ckpt["model"])
        self.target.load_state_dict(ckpt["model"])
        return ckpt.get("norms")
