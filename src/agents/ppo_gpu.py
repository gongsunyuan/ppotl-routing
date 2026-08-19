"""GPU PPO on BatchedActorCritic with S stacked seeds (ensemble) and vectorized updates."""
import numpy as np
import torch

from .batched_nets import BatchedActorCritic


class PPOGpuAgent:
    def __init__(self, state_dim, action_dim, cfg, device="cuda", use_pbrs=False,
                 freeze_fraction=0.0, seeds=(0,), minibatch=4096):
        self.cfg = cfg
        pd = cfg["paper_defaults"]
        self.device = torch.device(device)
        self.S = len(seeds)
        self.seeds = list(seeds)
        self.action_dim = action_dim
        self.model = BatchedActorCritic(state_dim, action_dim,
                                        hidden=pd["hidden_dim"], n_backbone=2,
                                        S=self.S, base_seed=seeds[0]).to(self.device)
        self.gamma = pd["gamma"]
        self.clip_eps = pd["clip_eps"]
        self.k_epochs = pd["k_epochs"]
        self.zeta1 = pd["zeta1"]
        self.zeta2 = pd["zeta2"]
        self.use_pbrs = use_pbrs
        self.freeze_fraction = freeze_fraction
        self.masks = self.model.freeze_grad_masks(freeze_fraction)
        self.minibatch = int(minibatch)
        actor_params, critic_params = [], []
        for name, p in self.model.named_parameters():
            (critic_params if name.startswith("critic") else actor_params).append(p)
        self.optimizer = torch.optim.Adam([
            {"params": actor_params, "lr": pd["actor_lr"]},
            {"params": critic_params, "lr": pd["critic_lr"]},
        ])

    # ---------------- rollout ----------------

    @torch.no_grad()
    def act(self, states):
        """states [N, D] -> actions [N], logps [N] (single-model rollout)."""
        logits, _ = self.model(states.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits[0])
        a = dist.sample()
        return a, dist.log_prob(a)

    @torch.no_grad()
    def act_ensemble(self, states):
        """states [S, R, D] -> actions [S, R], logps [S, R]."""
        logits, _ = self.model(states)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a)

    @torch.no_grad()
    def act_greedy(self, states):
        logits, _ = self.model(states.unsqueeze(0))
        return torch.argmax(logits[0], dim=-1)

    @torch.no_grad()
    def act_greedy_ensemble(self, states):
        logits, _ = self.model(states)
        return torch.argmax(logits, dim=-1)

    def shape_reward(self, r, s, s_next):
        if not self.use_pbrs:
            return r
        phi = lambda x: self.zeta1 * (1.0 - x[..., -2]) + self.zeta2 * (1.0 - x[..., -1])
        return r + self.gamma * phi(s_next) - phi(s)

    # ---------------- update ----------------

    def update(self, states, actions, logps, rewards, next_states):
        """Rollout tensors laid out seed-major: [T, N, ...] with N = S * R.

        Standard PPO: advantage/target computed once from pre-update values; clipped
        surrogate + half-squared critic loss over minibatches; one optimizer step per
        epoch with per-seed gradient accumulation (equivalent to summed per-seed losses).
        """
        T, N = states.shape[0], states.shape[1]
        R = N // self.S
        permute = lambda x: x.reshape(T, self.S, R, *x.shape[2:]).permute(1, 0, 2, *range(3, x.dim() + 1)) \
            .reshape(self.S, T * R, *x.shape[2:])
        st, ns = permute(states), permute(next_states)
        ac, lp = permute(actions).long(), permute(logps)
        rw = permute(rewards)
        Tn = T * R

        with torch.no_grad():
            _, v = self.model(st)
            _, vn = self.model(ns)
            delta = rw + self.gamma * vn - v
            target = rw + self.gamma * vn

        mb = min(self.minibatch, Tn)
        losses_a, losses_c = [], []
        for _ in range(self.k_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for s in range(self.S):
                perm = torch.randperm(Tn, device=self.device)
                for start in range(0, Tn, mb):
                    sel = perm[start:start + mb]
                    logits, values = self.model.forward_seed(st[s, sel], s)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(ac[s, sel])
                    ratio = torch.exp(logp - lp[s, sel])
                    surr1 = ratio * delta[s, sel]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * delta[s, sel]
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * ((target[s, sel] - values) ** 2).mean()
                    ((actor_loss + critic_loss) / self.S).backward()
                    losses_a.append(float(actor_loss.item()))
                    losses_c.append(float(critic_loss.item()))
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    p.grad.mul_(self.masks[name])
            self.optimizer.step()
        return float(np.mean(losses_a)), float(np.mean(losses_c))

    # ---------------- checkpoint ----------------

    def save(self, path, norms=None):
        states = [self.model.slice_state_dict(s) for s in range(self.S)]
        cpu_states = [self.model.to_cpu_state_dict(s) for s in range(self.S)]
        torch.save({"stacked": states, "stacked_cpu": cpu_states,
                    "seeds": self.seeds, "norms": norms}, path)

    def load(self, path):
        """Load stacked ckpt (seed-aligned when seeds match), or broadcast the first slice."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        stacked = ckpt.get("stacked")
        if stacked is None:
            stacked = [ckpt.get("model", ckpt)]
        with torch.no_grad():
            msd = self.model.state_dict()
            seeds = ckpt.get("seeds", list(range(len(stacked))))
            for si in range(self.S):
                want = self.seeds[si]
                if want in seeds:
                    src = stacked[seeds.index(want)]
                else:
                    src = stacked[si % len(stacked)]
                for k, v in src.items():
                    if k in msd:
                        msd[k][si].copy_(v.to(self.device))
        return ckpt.get("norms")
