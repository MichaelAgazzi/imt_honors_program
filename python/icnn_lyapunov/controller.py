from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

try:
    from motor import A_list, B_list
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from motor import A_list, B_list


@dataclass(frozen=True)
class ControlResult:
    u: np.ndarray
    gamma_current: float
    gamma_next: float
    lyapunov_delta: float
    stability_margin: float
    stable: bool
    n_iter: int


@dataclass(frozen=True)
class ThetaControlResult:
    u: np.ndarray
    theta: np.ndarray
    F: np.ndarray
    Q: np.ndarray
    Y: np.ndarray
    gamma_current: float
    gamma_next: float
    lyapunov_delta: float
    stability_margin: float
    stable: bool
    n_iter: int


def _as_2d_tensor(value: np.ndarray | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _project_u(u: torch.Tensor, u_max: float | np.ndarray | torch.Tensor | None) -> torch.Tensor:
    if u_max is None:
        return u
    bound = torch.as_tensor(u_max, dtype=u.dtype, device=u.device).abs()
    return torch.clamp(u, min=-bound, max=bound)


def _theta_to_q_y(theta: torch.Tensor, *, q_eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Map unconstrained theta to Q = L L^T + q_eps I and Y.

    theta = [l11_raw, l21, l22_raw, y1, y2] for the 2-state, 1-input motor.
    The softplus diagonal makes Q positive definite throughout optimization.
    """
    l11 = torch.nn.functional.softplus(theta[..., 0]) + q_eps
    l21 = theta[..., 1]
    l22 = torch.nn.functional.softplus(theta[..., 2]) + q_eps
    zeros = torch.zeros_like(l11)
    row1 = torch.stack([l11, zeros], dim=-1)
    row2 = torch.stack([l21, l22], dim=-1)
    l_mat = torch.stack([row1, row2], dim=-2)
    eye = torch.eye(2, dtype=theta.dtype, device=theta.device).expand(theta.shape[:-1] + (2, 2))
    q_mat = l_mat @ l_mat.transpose(-1, -2) + q_eps * eye
    y_mat = theta[..., 3:5].unsqueeze(-2)
    return q_mat, y_mat


def _gamma_from_model(
    model_icnn: torch.nn.Module,
    x: torch.Tensor,
    q_mat: torch.Tensor | None = None,
    y_mat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate gamma for either gamma(x) or gamma(x, theta) checkpoints."""
    input_dim = getattr(model_icnn, "input_dim", x.shape[-1])
    if input_dim == x.shape[-1]:
        return model_icnn(x)
    if input_dim == 7 and q_mat is not None and y_mat is not None:
        q_features = torch.stack(
            [q_mat[..., 0, 0], q_mat[..., 0, 1], q_mat[..., 1, 1]],
            dim=-1,
        )
        y_features = y_mat.squeeze(-2)
        return model_icnn(torch.cat([x, q_features, y_features], dim=-1))
    raise ValueError(
        f"Unsupported ICNN input_dim={input_dim}. Expected gamma(x) with input_dim=2 "
        "or gamma(x, theta) with input_dim=7=[x1,x2,q11,q12,q22,y1,y2]."
    )


def compute_control_action(
    x_k: np.ndarray | torch.Tensor,
    model_icnn: torch.nn.Module,
    *,
    A: np.ndarray | torch.Tensor | None = None,
    B: np.ndarray | torch.Tensor | None = None,
    F: np.ndarray | torch.Tensor | None = None,
    u_max: float | np.ndarray | torch.Tensor | None = None,
    epsilon: float = 1e-3,
    lr: float = 5e-2,
    n_steps: int = 120,
    stability_weight: float = 100.0,
    control_weight: float = 1e-4,
) -> ControlResult:
    """Optimize u so gamma_hat(Ax + Bu) decreases with a Lyapunov margin.

    The inner loop uses torch.autograd.grad explicitly to obtain
    d gamma_hat(x_next) / d u and the penalty gradients. Saturation is enforced
    after every gradient step by projection.
    """
    if A is None:
        A = A_list[0]
    if B is None:
        B = B_list[0]

    device = next(model_icnn.parameters()).device
    was_training = model_icnn.training
    model_icnn.eval()

    x = _as_2d_tensor(x_k, device=device)
    A_t = torch.as_tensor(A, dtype=torch.float32, device=device)
    B_t = torch.as_tensor(B, dtype=torch.float32, device=device)
    if A_t.ndim != 2 or B_t.ndim != 2:
        raise ValueError("A and B must be two-dimensional matrices.")

    state_dim = A_t.shape[0]
    input_dim = B_t.shape[1]
    if x.shape[1] != state_dim:
        raise ValueError(f"x_k has dimension {x.shape[1]}, expected {state_dim}.")

    if F is not None:
        F_t = torch.as_tensor(F, dtype=torch.float32, device=device)
        u = x @ F_t.T if F_t.shape[-1] == state_dim else x @ F_t
    else:
        u = torch.zeros((x.shape[0], input_dim), dtype=torch.float32, device=device)
    u = _project_u(u, u_max).detach().clone().requires_grad_(True)

    with torch.no_grad():
        gamma_current = model_icnn(x)
        margin = epsilon * torch.sum(x * x, dim=1, keepdim=True)

    best_u = u.detach().clone()
    best_loss = torch.inf
    best_gamma_next = None

    for step in range(1, n_steps + 1):
        x_next = x @ A_t.T + u @ B_t.T
        gamma_next = model_icnn(x_next)
        lyapunov_residual = gamma_next - gamma_current + margin
        stability_penalty = torch.relu(lyapunov_residual).pow(2).mean()
        control_penalty = u.pow(2).mean()
        loss = gamma_next.mean() + stability_weight * stability_penalty + control_weight * control_penalty

        grad_gamma_u = torch.autograd.grad(gamma_next.mean(), u, retain_graph=True)[0]
        grad_stability_u = torch.autograd.grad(stability_penalty, u, retain_graph=True)[0]
        grad_control_u = torch.autograd.grad(control_penalty, u, create_graph=False)[0]
        grad_u = grad_gamma_u + stability_weight * grad_stability_u + control_weight * grad_control_u
        with torch.no_grad():
            candidate_u = _project_u(u - lr * grad_u, u_max)
            if loss.detach() < best_loss:
                best_loss = loss.detach()
                best_u = u.detach().clone()
                best_gamma_next = gamma_next.detach().clone()
            u.copy_(candidate_u)
        u.requires_grad_(True)

    with torch.no_grad():
        x_next = x @ A_t.T + best_u @ B_t.T
        final_gamma_next = model_icnn(x_next)
        if best_gamma_next is None or final_gamma_next.mean() < best_gamma_next.mean():
            best_gamma_next = final_gamma_next
        delta = best_gamma_next - gamma_current
        stable = bool(torch.all(delta < -margin).item())

    if was_training:
        model_icnn.train()

    return ControlResult(
        u=best_u.detach().cpu().numpy().reshape(-1),
        gamma_current=float(gamma_current.detach().cpu().mean().item()),
        gamma_next=float(best_gamma_next.detach().cpu().mean().item()),
        lyapunov_delta=float(delta.detach().cpu().mean().item()),
        stability_margin=float(margin.detach().cpu().mean().item()),
        stable=stable,
        n_iter=n_steps,
    )


def compute_control_action_theta(
    x_k: np.ndarray | torch.Tensor,
    model_icnn: torch.nn.Module,
    *,
    A: np.ndarray | torch.Tensor | None = None,
    B: np.ndarray | torch.Tensor | None = None,
    u_max: float | np.ndarray | torch.Tensor | None = None,
    beta: float = 1e-4,
    lr: float = 5e-2,
    n_steps: int = 250,
    stability_weight: float = 500.0,
    state_weight: float = 1.0,
    control_weight: float = 1e-4,
    q_eps: float = 1e-3,
    theta_init: np.ndarray | torch.Tensor | None = None,
) -> ThetaControlResult:
    """Optimize theta=(Q,Y), then apply u = Y Q^{-1} x_k.

    The Lyapunov margin is B = beta * ||x_k||^2. If the checkpoint was trained
    as gamma(x, theta), gamma is evaluated on [x, Q, Y]. If it was trained as
    gamma(x), theta affects the objective through x_{k+1}=Ax_k+Bu_k.
    """
    if A is None:
        A = A_list[0]
    if B is None:
        B = B_list[0]

    device = next(model_icnn.parameters()).device
    was_training = model_icnn.training
    model_icnn.eval()

    x = _as_2d_tensor(x_k, device=device)
    A_t = torch.as_tensor(A, dtype=torch.float32, device=device)
    B_t = torch.as_tensor(B, dtype=torch.float32, device=device)
    if x.shape[1] != 2:
        raise ValueError("compute_control_action_theta currently expects a 2D state.")

    if theta_init is None:
        theta = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    else:
        theta = torch.as_tensor(theta_init, dtype=torch.float32, device=device).reshape(1, 5)
    theta = theta.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    best = None
    best_loss = torch.inf

    for step in range(1, n_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        q_mat, y_mat = _theta_to_q_y(theta, q_eps=q_eps)
        f_mat = torch.linalg.solve(q_mat, y_mat.transpose(-1, -2)).transpose(-1, -2)
        u = torch.bmm(f_mat, x.unsqueeze(-1)).squeeze(-1)
        u = _project_u(u, u_max)
        x_next = x @ A_t.T + u @ B_t.T

        gamma_current = _gamma_from_model(model_icnn, x, q_mat, y_mat)
        gamma_next = _gamma_from_model(model_icnn, x_next, q_mat, y_mat)
        margin = beta * torch.sum(x * x, dim=1, keepdim=True)
        lyapunov_residual = gamma_next - gamma_current + margin
        stability_penalty = torch.relu(lyapunov_residual).pow(2).mean()
        q_regularization = (q_mat - torch.eye(2, dtype=q_mat.dtype, device=q_mat.device)).pow(2).mean()
        loss = (
            gamma_next.mean()
            + stability_weight * stability_penalty
            + state_weight * x_next.pow(2).mean()
            + control_weight * u.pow(2).mean()
            + 1e-5 * q_regularization
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if loss.detach() < best_loss:
                best_loss = loss.detach().clone()
                best = (
                    theta.detach().clone(),
                    q_mat.detach().clone(),
                    y_mat.detach().clone(),
                    f_mat.detach().clone(),
                    u.detach().clone(),
                    gamma_current.detach().clone(),
                    gamma_next.detach().clone(),
                    margin.detach().clone(),
                )

    if best is None:
        raise RuntimeError("Theta optimization did not produce a candidate.")

    theta_best, q_best, y_best, f_best, u_best, gamma_current_best, gamma_next_best, margin_best = best
    delta = gamma_next_best - gamma_current_best
    stable = bool(torch.all(delta < -margin_best).item())

    if was_training:
        model_icnn.train()

    return ThetaControlResult(
        u=u_best.detach().cpu().numpy().reshape(-1),
        theta=theta_best.detach().cpu().numpy().reshape(5),
        F=f_best.detach().cpu().numpy().reshape(1, 2),
        Q=q_best.detach().cpu().numpy().reshape(2, 2),
        Y=y_best.detach().cpu().numpy().reshape(1, 2),
        gamma_current=float(gamma_current_best.detach().cpu().mean().item()),
        gamma_next=float(gamma_next_best.detach().cpu().mean().item()),
        lyapunov_delta=float(delta.detach().cpu().mean().item()),
        stability_margin=float(margin_best.detach().cpu().mean().item()),
        stable=stable,
        n_iter=n_steps,
    )
