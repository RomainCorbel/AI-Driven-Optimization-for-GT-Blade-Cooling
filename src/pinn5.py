import torch
import torch.nn as nn
from constants import *
from utils import *


class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(PINNs, self).__init__()

        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))
        self.linear = nn.Sequential(*layers)

    def forward(self, train_points):
        return self.linear(train_points)


class NormalizedPINNs(nn.Module):
    """Wrapper that z-scores inputs and unstandardizes outputs.

    The inner PINN sees O(1) inputs and is trained to predict O(1) outputs.
    forward() returns physical-unit quantities so get_values_and_derivatives
    and get_loss need no changes. Autograd chain rule through both affine maps
    keeps all spatial derivatives in physical units.
    """

    def __init__(self, pinn, coord_mean, coord_std, out_mean, out_std):
        super().__init__()
        self.pinn = pinn
        self.register_buffer("coord_mean", coord_mean)
        self.register_buffer("coord_std", coord_std)
        self.register_buffer("out_mean", out_mean)
        self.register_buffer("out_std", out_std)

    def forward(self, x):
        x_norm   = (x - self.coord_mean) / self.coord_std
        out_norm = self.pinn(x_norm)
        safe_std = torch.where(self.out_std == 0, torch.ones_like(self.out_std), self.out_std)
        return out_norm * safe_std + self.out_mean


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def make_loss_weights(n, device):
    """Return learnable adaptive loss weights initialized to 1.0."""
    return nn.Parameter(torch.ones(n, dtype=torch.float64, device=device))


def get_values_and_derivatives(fields, points):
    vx = fields[:, 0:1]
    vy = fields[:, 1:2]
    vz = fields[:, 2:3]
    p  = fields[:, 3:4]
    T  = fields[:, 4:5] * 1000

    # --- vx ---
    grad_vx = torch.autograd.grad(
        vx, points, torch.ones_like(vx), retain_graph=True, create_graph=True
    )[0]
    vx_x = grad_vx[:, 0:1]
    vx_y = grad_vx[:, 1:2]
    vx_z = grad_vx[:, 2:3]
    vx_xx = torch.autograd.grad(vx_x, points, torch.ones_like(vx_x), retain_graph=True, create_graph=True)[0][:, 0:1]
    vx_yy = torch.autograd.grad(vx_y, points, torch.ones_like(vx_y), retain_graph=True, create_graph=True)[0][:, 1:2]
    vx_zz = torch.autograd.grad(vx_z, points, torch.ones_like(vx_z), retain_graph=True, create_graph=True)[0][:, 2:3]

    # --- vy ---
    grad_vy = torch.autograd.grad(
        vy, points, torch.ones_like(vy), retain_graph=True, create_graph=True
    )[0]
    vy_x = grad_vy[:, 0:1]
    vy_y = grad_vy[:, 1:2]
    vy_z = grad_vy[:, 2:3]
    vy_xx = torch.autograd.grad(vy_x, points, torch.ones_like(vy_x), retain_graph=True, create_graph=True)[0][:, 0:1]
    vy_yy = torch.autograd.grad(vy_y, points, torch.ones_like(vy_y), retain_graph=True, create_graph=True)[0][:, 1:2]
    vy_zz = torch.autograd.grad(vy_z, points, torch.ones_like(vy_z), retain_graph=True, create_graph=True)[0][:, 2:3]

    # --- vz ---
    grad_vz = torch.autograd.grad(
        vz, points, torch.ones_like(vz), retain_graph=True, create_graph=True
    )[0]
    vz_x = grad_vz[:, 0:1]
    vz_y = grad_vz[:, 1:2]
    vz_z = grad_vz[:, 2:3]
    vz_xx = torch.autograd.grad(vz_x, points, torch.ones_like(vz_x), retain_graph=True, create_graph=True)[0][:, 0:1]
    vz_yy = torch.autograd.grad(vz_y, points, torch.ones_like(vz_y), retain_graph=True, create_graph=True)[0][:, 1:2]
    vz_zz = torch.autograd.grad(vz_z, points, torch.ones_like(vz_z), retain_graph=True, create_graph=True)[0][:, 2:3]

    # --- p (first-order only) ---
    grad_p = torch.autograd.grad(
        p, points, torch.ones_like(p), retain_graph=True, create_graph=True
    )[0]
    p_x = grad_p[:, 0:1]
    p_y = grad_p[:, 1:2]
    p_z = grad_p[:, 2:3]

    # --- T ---
    # T = fields[:,4:5] * 1000, so grad_T already carries the ×1000 factor.
    grad_T = torch.autograd.grad(
        T, points, torch.ones_like(T), retain_graph=True, create_graph=True
    )[0]
    T_x = grad_T[:, 0:1]
    T_y = grad_T[:, 1:2]
    T_z = grad_T[:, 2:3]
    T_xx = torch.autograd.grad(T_x, points, torch.ones_like(T_x), retain_graph=True, create_graph=True)[0][:, 0:1]
    T_yy = torch.autograd.grad(T_y, points, torch.ones_like(T_y), retain_graph=True, create_graph=True)[0][:, 1:2]
    T_zz = torch.autograd.grad(T_z, points, torch.ones_like(T_z), retain_graph=True, create_graph=True)[0][:, 2:3]

    return (
        vx, vy, vz, p, T,
        vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
        vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
        vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
        p_x, p_y, p_z,
        T_x, T_y, T_z, T_xx, T_yy, T_zz,
    )


def get_loss(
    vx, vy, vz, p, T,
    vx_x, vx_y, vx_z, vx_xx, vx_yy, vx_zz,
    vy_x, vy_y, vy_z, vy_xx, vy_yy, vy_zz,
    vz_x, vz_y, vz_z, vz_xx, vz_yy, vz_zz,
    p_x, p_y, p_z,
    T_x, T_y, T_z, T_xx, T_yy, T_zz,
    labels,
    p_outlet,
):
    loss_divergence = torch.mean((vx_x + vy_y + vz_z) ** 2)

    mu = dynamic_viscosity(T)

    loss_momentum_x = torch.mean(
        (
            vx[labels == 0] * vx_x[labels == 0]
            + vy[labels == 0] * vx_y[labels == 0]
            + vz[labels == 0] * vx_z[labels == 0]
            + (10**5 / rho) * p_x[labels == 0]
            - (mu[labels == 0] / rho)
            * (vx_xx[labels == 0] + vx_yy[labels == 0] + vx_zz[labels == 0])
        )
        ** 2
    )
    loss_momentum_y = torch.mean(
        (
            vx[labels == 0] * vy_x[labels == 0]
            + vy[labels == 0] * vy_y[labels == 0]
            + vz[labels == 0] * vy_z[labels == 0]
            + (10**5 / rho) * p_y[labels == 0]
            - (mu[labels == 0] / rho)
            * (vy_xx[labels == 0] + vy_yy[labels == 0] + vy_zz[labels == 0])
        )
        ** 2
    )
    loss_momentum_z = torch.mean(
        (
            vx[labels == 0] * vz_x[labels == 0]
            + vy[labels == 0] * vz_y[labels == 0]
            + vz[labels == 0] * vz_z[labels == 0]
            + (10**5 / rho) * p_z[labels == 0]
            - (mu[labels == 0] / rho)
            * (vz_xx[labels == 0] + vz_yy[labels == 0] + vz_zz[labels == 0])
        )
        ** 2
    )

    loss_other_boundary = (
        torch.mean((vx[labels == 2]) ** 2)
        + torch.mean((vy[labels == 2]) ** 2)
        + torch.mean((vz[labels == 2]) ** 2)
    )

    alpha = thermal_conductivity / (density * specific_heat)
    loss_heat = (
        torch.mean(
            (
                alpha * (T_xx[labels == 0] + T_yy[labels == 0] + T_zz[labels == 0])
                + vx[labels == 0] * T_x[labels == 0]
                + vy[labels == 0] * T_y[labels == 0]
                + vz[labels == 0] * T_z[labels == 0]
            )
            ** 2
        )
        / 10**6
    )

    loss_t_wall_boundary = torch.mean((T[labels == 2] - T_wall) ** 2) / 10**6

    loss_outlet_boundary = torch.mean((p[labels == 3] - p_outlet) ** 2)

    return (
        loss_divergence,
        loss_momentum_x,
        loss_momentum_y,
        loss_momentum_z,
        loss_outlet_boundary,
        loss_other_boundary,
        loss_heat,
        loss_t_wall_boundary,
    )
