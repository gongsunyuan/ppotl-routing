"""Batched actor-critic: S independent models stacked as [S, ...] tensors.

A single forward processes [S, B, D] inputs via broadcast matmul, so S seeds train
simultaneously (L3 ensemble). Because Adam updates are elementwise, one optimizer over
stacked parameters is exactly equivalent to S independent optimizers.

Parameter names are flat (torch forbids '.' in registered parameter names); the mapping
to CPU ActorCritic names is done in to_cpu_state_dict for checkpoint interoperability.
"""
import math
import zlib

import torch
import torch.nn as nn

PARAM_NAMES = ["b0_w", "b0_b", "b1_w", "b1_b", "a_w", "a_b", "c_w", "c_b"]
CPU_NAME_MAP = {
    "b0_w": "backbone.0.weight", "b0_b": "backbone.0.bias",
    "b1_w": "backbone.1.weight", "b1_b": "backbone.1.bias",
    "a_w": "actor_head.weight", "a_b": "actor_head.bias",
    "c_w": "critic_head.weight", "c_b": "critic_head.bias",
}


def _det_hash(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


class BatchedActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, n_backbone=2, S=1, base_seed=0):
        super().__init__()
        self.S = S
        self.n_backbone = n_backbone
        self.state_dim, self.action_dim, self.hidden = state_dim, action_dim, hidden
        shapes = {
            "b0_w": (S, state_dim, hidden), "b0_b": (S, hidden),
            "b1_w": (S, hidden, hidden), "b1_b": (S, hidden),
            "a_w": (S, hidden, action_dim), "a_b": (S, action_dim),
            "c_w": (S, hidden, 1), "c_b": (S, 1),
        }
        for name, shape in shapes.items():
            self.register_parameter(name, nn.Parameter(torch.empty(*shape)))
        self.reset_parameters(base_seed)

    def reset_parameters(self, base_seed=0):
        with torch.no_grad():
            for name, p in self.named_parameters():
                fan_in = p.shape[1]
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0
                for s in range(self.S):
                    torch.manual_seed(_det_hash(name, base_seed + s))
                    p.data[s].uniform_(-bound, bound)

    def forward(self, x):
        """x: [S, B, D] -> logits [S, B, A], values [S, B]."""
        P = dict(self.named_parameters())
        h = torch.tanh(torch.baddbmm(P["b0_b"].unsqueeze(1), x, P["b0_w"]))
        if "b1_w" in P:
            h = torch.tanh(torch.baddbmm(P["b1_b"].unsqueeze(1), h, P["b1_w"]))
        logits = torch.baddbmm(P["a_b"].unsqueeze(1), h, P["a_w"])
        values = torch.baddbmm(P["c_b"].unsqueeze(1), h, P["c_w"]).squeeze(-1)
        return logits, values

    def forward_seed(self, x, s):
        """Single-seed forward for any S: x [B, D] -> logits [B, A], values [B]."""
        P = dict(self.named_parameters())
        h = torch.tanh(torch.addmm(P["b0_b"][s], x, P["b0_w"][s]))
        if "b1_w" in P:
            h = torch.tanh(torch.addmm(P["b1_b"][s], h, P["b1_w"][s]))
        logits = torch.addmm(P["a_b"][s], h, P["a_w"][s])
        values = torch.addmm(P["c_b"][s], h, P["c_w"][s]).squeeze(-1)
        return logits, values

    def freeze_grad_masks(self, freeze_fraction):
        """Masks matching CPU semantics: freeze round(frac * n_backbone) leading layers."""
        n_freeze = int(round(freeze_fraction * self.n_backbone))
        self._n_freeze = n_freeze
        masks = {}
        for name, p in self.named_parameters():
            m = torch.ones_like(p)
            if name.startswith("b0") and n_freeze >= 1:
                m.zero_()
            if name.startswith("b1") and n_freeze >= 2:
                m.zero_()
            masks[name] = m
        return masks

    def frozen_parameter_count(self):
        total = sum(p.numel() for p in self.parameters())
        frozen = 0
        n_freeze = getattr(self, "_n_freeze", 0)
        for name, p in self.named_parameters():
            if (name.startswith("b0") and n_freeze >= 1) or (name.startswith("b1") and n_freeze >= 2):
                frozen += p.numel()
        return frozen, total

    def slice_state_dict(self, s):
        return {k: v[s].cpu().clone() for k, v in self.state_dict().items()}

    def to_cpu_state_dict(self, s):
        """Convert slice s into CPU ActorCritic naming for cross-backend checkpoints."""
        sd = self.slice_state_dict(s)
        return {CPU_NAME_MAP[k]: v for k, v in sd.items()}
