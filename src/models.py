"""
models.py — neural network architectures shared across PINN and QPP-MLP pipelines.
"""

import torch
import torch.nn as nn
from config import WALL_Y1, WALL_Y2, WALL_EPS


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class FFNN(nn.Module):
    """Generic feedforward network with Sin activations and Xavier initialization."""
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), Sin()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        for lin in [m for m in self.net if isinstance(m, nn.Linear)]:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        return self.net(x)


class NormalizedPINN(nn.Module):
    """Wraps FFNN with z-scored coordinate inputs and un-standardized outputs.

    Input layout expected by forward(): [x, y, z, AR_nd, EDH_nd, PE_nd, ALPHA_nd, RE]  (8 dims).
    The forward pass appends two sigmoid wall-distance features computed from x[:,1] (physical y)
    before z-scoring, giving a 10-dim input to the underlying net.
    coord_mean/coord_std must cover all 10 dims.
    """
    def __init__(self, net, coord_mean, coord_std, out_mean, out_std):
        super().__init__()
        self.net = net
        self.register_buffer("coord_mean", coord_mean)
        self.register_buffer("coord_std",  coord_std)
        self.register_buffer("out_mean",   out_mean)
        self.register_buffer("out_std",    out_std)

    def forward(self, x):
        # x[:,1] is physical y — append wall sigmoid features before z-scoring
        y   = x[:, 1:2]
        s1  = torch.sigmoid((y - WALL_Y1) / WALL_EPS)
        s2  = torch.sigmoid((y - WALL_Y2) / WALL_EPS)
        x   = torch.cat([x, s1, s2], dim=1)
        x_norm   = (x - self.coord_mean) / self.coord_std
        y_norm   = self.net(x_norm)
        safe_std = torch.where(self.out_std == 0, torch.ones_like(self.out_std), self.out_std)
        return y_norm * safe_std + self.out_mean


class QPP_MLP(nn.Module):
    """Wall heat-flux surrogate: (x, y, , AR_nd, EDH_nd, PE_nd, ALPHA_nd, RE, s1, s2) → q'' [normalised scalar]."""
    def __init__(self, in_dim, hidden_dim, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), Sin()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (N,)
