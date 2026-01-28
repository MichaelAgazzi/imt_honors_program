from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class GammaICNN(nn.Module):
    """Fully input-convex neural network for approximating gamma*(x).

    This follows the Amos et al. ICNN structure

        z_{k+1} = relu(W_z z_k + W_x x + b)
        y       = W_z^out z_L + W_x^out x + b

    Convexity with respect to the input is preserved by keeping all W_z blocks
    non-negative. The W_x terms are affine in x and can remain unconstrained.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: Sequence[int] = (64, 64, 32),
        activation: str = "softplus",
        enforce_positive_output: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")

        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation = activation
        self.enforce_positive_output = enforce_positive_output

        self.x_layers = nn.ModuleList()
        self.z_layers = nn.ModuleList()

        for layer_id, width in enumerate(self.hidden_sizes):
            self.x_layers.append(nn.Linear(self.input_dim, width))
            if layer_id > 0:
                self.z_layers.append(
                    nn.Linear(self.hidden_sizes[layer_id - 1], width, bias=False)
                )

        self.x_out = nn.Linear(self.input_dim, 1)
        self.z_out = nn.Linear(self.hidden_sizes[-1], 1, bias=False)

        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_std", torch.ones(self.input_dim))
        self.register_buffer("lower_bound_offset", torch.zeros(1))

        self.reset_parameters()
        self.project_nonnegative_weights()

    def reset_parameters(self) -> None:
        for layer in self.x_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.z_layers:
            nn.init.uniform_(layer.weight, a=0.0, b=0.1)
        nn.init.xavier_uniform_(self.x_out.weight)
        nn.init.zeros_(self.x_out.bias)
        nn.init.uniform_(self.z_out.weight, a=0.0, b=0.1)

    def set_input_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.input_dim or std.numel() != self.input_dim:
            raise ValueError("Normalization vectors must match input_dim.")
        std = torch.clamp(std.detach().float().flatten(), min=1e-8)
        self.input_mean.copy_(mean.detach().float().flatten())
        self.input_std.copy_(std)

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(value)
        if self.activation == "softplus":
            return F.softplus(value)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = (x - self.input_mean) / self.input_std

        z = self._activate(self.x_layers[0](x))
        for x_layer, z_layer in zip(self.x_layers[1:], self.z_layers):
            z = self._activate(x_layer(x) + z_layer(z))

        gamma = self.x_out(x) + self.z_out(z)
        if self.enforce_positive_output:
            gamma = F.softplus(gamma)
        return gamma - self.lower_bound_offset

    @torch.no_grad()
    def project_nonnegative_weights(self) -> None:
        """Project ICNN recurrent/output weights onto the non-negative cone."""
        for layer in self.z_layers:
            layer.weight.clamp_(min=0.0)
        self.z_out.weight.clamp_(min=0.0)

    @torch.no_grad()
    def calibrate_lower_bound(
        self,
        x: torch.Tensor,
        gamma_true: torch.Tensor,
        safety_margin: float = 1e-6,
    ) -> float:
        """Shift predictions down so the finite calibration set is not overfit above gamma.

        A constant downward shift preserves convexity. This is a dataset-level
        calibration step and should not be interpreted as a formal certificate
        outside the sampled domain.
        """
        pred = self(x)
        violation = torch.clamp(pred - gamma_true, min=0.0).max()
        shift = float(violation.item() + safety_margin)
        self.lower_bound_offset.add_(shift)
        return shift


class ICNNLyapunov(nn.Module):
    """Positive-definite Lyapunov/value approximator built from an ICNN.

    The raw ICNN ``phi_theta`` is convex in its input. The public Lyapunov
    prediction is

        V_hat(x) = phi_theta(x) - phi_theta(0) + eps * ||x||^2,

    which pins the origin to zero and adds a configurable quadratic term.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: Sequence[int] = (64, 64, 32),
        activation: str = "softplus",
        eps_positive_definite: float = 1e-4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.eps_positive_definite = float(eps_positive_definite)
        self.phi = GammaICNN(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            enforce_positive_output=False,
        )

    def set_input_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.phi.set_input_normalization(mean, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.V_hat(x)

    def V_hat(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        zeros = torch.zeros_like(x)
        return self.phi(x) - self.phi(zeros) + self.eps_positive_definite * torch.sum(
            x * x,
            dim=1,
            keepdim=True,
        )

    @torch.no_grad()
    def project_nonnegative_weights(self) -> None:
        self.phi.project_nonnegative_weights()


class ActorNetwork(nn.Module):
    """Feed-forward actor ``u = pi_theta(x)`` for the motor controller."""

    def __init__(
        self,
        state_dim: int = 2,
        control_dim: int = 1,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        u_max: float | None = None,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or control_dim <= 0:
            raise ValueError("state_dim and control_dim must be positive.")

        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation = activation
        self.u_max = None if u_max is None else float(abs(u_max))
        self.residual_scale = float(residual_scale)
        self.register_buffer("linear_skip", torch.zeros(self.control_dim, self.state_dim))

        layers: list[nn.Module] = []
        in_dim = self.state_dim
        for width in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, width))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "softplus":
                layers.append(nn.Softplus())
            else:
                raise ValueError(f"Unsupported actor activation: {activation}")
            in_dim = width
        layers.append(nn.Linear(in_dim, self.control_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        u_linear = x @ self.linear_skip.T
        u = u_linear + self.residual_scale * self.net(x)
        if self.u_max is not None:
            u = self.u_max * torch.tanh(u / max(self.u_max, 1e-8))
        return u

    @torch.no_grad()
    def set_linear_skip(self, gain: torch.Tensor) -> None:
        gain = torch.as_tensor(gain, dtype=self.linear_skip.dtype, device=self.linear_skip.device)
        if gain.shape != self.linear_skip.shape:
            raise ValueError(f"linear skip gain must have shape {tuple(self.linear_skip.shape)}, got {tuple(gain.shape)}.")
        self.linear_skip.copy_(gain)
