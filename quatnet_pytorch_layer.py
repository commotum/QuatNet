import math
import torch
import torch.nn as nn


def hamilton_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the Hamilton product between two tensors of quaternions.

    Both ``a`` and ``b`` must have the same shape and last dimension size 4.
    Broadcasting is supported on the leading dimensions.
    Returns a tensor of the same broadcasted shape with last dimension 4.
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((rw, rx, ry, rz), dim=-1)


def apply_activation(x: torch.Tensor, kind: str) -> torch.Tensor:
    kind = kind.lower()
    if kind == "split_tanh":
        return torch.tanh(x)
    if kind == "split_sigmoid":
        return torch.sigmoid(x)
    if kind == "pure_imag_sigmoid":
        real = torch.zeros_like(x[..., :1])
        imag = torch.sigmoid(x[..., 1:])
        return torch.cat((real, imag), dim=-1)
    return x


class QuatNetPytorchDenseLayer(nn.Module):
    """Pure PyTorch quaternion dense layer using Hamilton products."""

    def __init__(self, input_dim_q: int, output_dim_q: int, activation: str = "none"):
        super().__init__()
        self.input_dim_q = input_dim_q
        self.output_dim_q = output_dim_q
        self.activation = activation

        self.W = nn.Parameter(torch.empty(output_dim_q, input_dim_q, 4))
        self.B = nn.Parameter(torch.zeros(output_dim_q, 4))
        self.quaternion_init()

    def quaternion_init(self):
        """Initialize ``self.W`` using Parcollet et al. method."""
        n_in = float(self.input_dim_q)
        n_out = float(self.output_dim_q)
        sigma = 1.0 / (2.0 * (n_in + n_out)) ** 0.5

        device = self.W.device
        theta = (2 * math.pi) * torch.rand(self.W.shape[0], self.W.shape[1], device=device) - math.pi
        phi = (2 * sigma) * torch.rand(self.W.shape[0], self.W.shape[1], device=device) - sigma
        xyz = torch.rand(self.W.shape[0], self.W.shape[1], 3, device=device)
        norm = torch.norm(xyz, dim=2, keepdim=True) + 1e-12
        u = xyz / norm
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        self.W.data[..., 0] = phi * cos_t
        self.W.data[..., 1] = phi * u[..., 0] * sin_t
        self.W.data[..., 2] = phi * u[..., 1] * sin_t
        self.W.data[..., 3] = phi * u[..., 2] * sin_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim_q, 4)
        w = self.W.unsqueeze(0)            # (1, out, in, 4)
        x_exp = x.unsqueeze(1)             # (batch, 1, in, 4)
        prod = hamilton_product(w, x_exp)  # (batch, out, in, 4)
        s = prod.sum(dim=2)                # (batch, out, 4)
        s = s + self.B
        return apply_activation(s, self.activation)

    def __repr__(self):
        return f"QuatNetPytorchDenseLayer(input_dim_q={self.input_dim_q}, output_dim_q={self.output_dim_q}, activation={self.activation})"
