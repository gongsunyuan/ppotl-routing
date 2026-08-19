import numpy as np
import torch
import torch.nn as nn
from .nets import ActorCritic


class PPOAgent:
    needs_pretrain = True

    def __init__(self, state_dim, action_dim, cfg, device="cpu",
                 use_pbrs=False, freeze_fraction=0.0, seed=0):
        torch.manual_seed(seed)
        self.cfg = cfg
        h = cfg["paper_defaults"]["hidden_dim"]
        self.model = ActorCritic(state_dim, action_dim, hidden=h).to(device)
        pd = cfg["paper_defaults"]
        self.gamma = pd["gamma"]
        self.clip_eps = pd["clip_eps"]
        self.k_epochs = pd["k_epochs"]
        self.actor_lr = pd["actor_lr"]
        self.critic_lr = pd["critic_lr"]
        self.zeta1 = pd["zeta1"]
        self.zeta2 = pd["zeta2"]
        self.use_pbrs = use_pbrs
        self.freeze_fraction = freeze_fraction
        if freeze_fraction > 0:
            self.model.set_frozen_fraction(freeze_fraction)
        actor_params = [p for p in self.model.actor_head.parameters() if p.requires_grad]
        backbone_actor = [p for p in self.model.backbone.parameters() if p.requires_grad]
        critic_params = list(self.model.critic_head.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": actor_params + backbone_actor, "lr": self.actor_lr},
            {"params": critic_params, "lr": self.critic_lr},
        ])

    def select_action(self, state, greedy=False):
        st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.model.act(st)
        if greedy:
            a = int(torch.argmax(dist.logits).item())
            return a, 0.0
        a = int(dist.sample().item())
        return a, float(dist.log_prob(torch.tensor(a)).item())

    def shape_reward(self, r, s, s_next):
        if not self.use_pbrs:
            return r
        phi_s = self.zeta1 * (1.0 - s[-2]) + self.zeta2 * (1.0 - s[-1])
        phi_ns = self.zeta1 * (1.0 - s_next[-2]) + self.zeta2 * (1.0 - s_next[-1])
        return r + self.gamma * phi_ns - phi_s

    def update(self, states, actions, logprobs, rewards, next_states):
        S = torch.as_tensor(np.array(states), dtype=torch.float32)
        A = torch.as_tensor(np.array(actions), dtype=torch.long)
        LP = torch.as_tensor(np.array(logprobs), dtype=torch.float32)
        R = torch.as_tensor(np.array(rewards), dtype=torch.float32)
        SN = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        actor_losses, critic_losses = [], []
        for _ in range(self.k_epochs):
            with torch.no_grad():
                _, v = self.model.act(S)
                _, vn = self.model.act(SN)
            delta = R + self.gamma * vn.squeeze(-1) - v.squeeze(-1)
            for idx in torch.randperm(len(S)):
                s_i = S[idx].unsqueeze(0)
                a_i = A[idx].unsqueeze(0)
                dist, value = self.model.act(s_i)
                logp = dist.log_prob(a_i)
                ratio = torch.exp(logp - LP[idx])
                surr1 = ratio * delta[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * delta[idx]
                actor_loss = -torch.min(surr1, surr2)
                target = R[idx] + self.gamma * vn[idx]
                critic_loss = 0.5 * (target - value.squeeze(-1)) ** 2
                self.optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                self.optimizer.step()
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
        return float(np.mean(actor_losses)), float(np.mean(critic_losses))

    def save(self, path, norms=None):
        torch.save({"model": self.model.state_dict(), "norms": norms}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        return ckpt.get("norms")
