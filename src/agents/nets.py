import torch
import torch.nn as nn


def build_mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
    )


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, n_backbone_layers=2):
        super().__init__()
        self.n_backbone_layers = n_backbone_layers
        layers = [nn.Linear(state_dim, hidden), nn.Tanh()]
        for _ in range(n_backbone_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden, action_dim)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, state):
        h = self.backbone(state)
        return self.actor_head(h), self.critic_head(h)

    def act(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value

    def set_frozen_fraction(self, fraction):
        n = self.n_backbone_layers
        n_freeze = int(round(fraction * n))
        for i, module in enumerate(self.backbone):
            is_linear = isinstance(module, nn.Linear)
            belongs = (i // 2) < n_freeze
            for p in module.parameters():
                p.requires_grad = not (is_linear and belongs)

    def frozen_parameter_count(self):
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return frozen, total


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state):
        return self.net(state)
