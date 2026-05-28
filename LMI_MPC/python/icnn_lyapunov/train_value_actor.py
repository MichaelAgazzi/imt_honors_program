from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from icnn_lyapunov.data import make_value_actor_dataloaders
    from icnn_lyapunov.models import ActorNetwork, ICNNLyapunov
else:
    from .data import make_value_actor_dataloaders
    from .models import ActorNetwork, ICNNLyapunov

try:
    from motor import A_list, B_list
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from motor import A_list, B_list


@dataclass(frozen=True)
class BetaBounds:
    beta_minus_emp: float
    beta_plus_emp: float
    beta_minus_cert: float
    beta_plus_cert: float
    beta_symmetric_emp: float
    beta_symmetric_cert: float
    L_V_emp: float
    L_V_cert: float
    L_Vhat_cert: float
    rho: float


@dataclass
class TrainingHistory:
    epoch: list[int]
    train_total: list[float]
    train_value: list[float]
    train_actor: list[float]
    train_decrease: list[float]
    train_rollout: list[float]
    val_total: list[float]
    val_value: list[float]
    val_actor: list[float]
    val_decrease: list[float]
    val_rollout: list[float]
    beta_minus_cert: list[float]
    beta_plus_cert: list[float]

    @classmethod
    def empty(cls) -> "TrainingHistory":
        return cls([], [], [], [], [], [], [], [], [], [], [], [], [])


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _project_root() / resolved
    return resolved


class MotorDynamics(nn.Module):
    """Batch motor dynamics ``x_next = A x + B u``."""

    def __init__(
        self,
        *,
        vertex_id: int = 1,
        u_max: float | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.vertex_id = int(vertex_id)
        self.u_max = None if u_max is None else float(abs(u_max))
        self.register_buffer("A", torch.as_tensor(A_list[self.vertex_id], dtype=torch.float32, device=device))
        self.register_buffer("B", torch.as_tensor(B_list[self.vertex_id], dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return dynamics_step(x, u, A=self.A, B=self.B, u_max=self.u_max)


def dynamics_step(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    u_max: float | None = None,
) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if u.ndim == 1:
        u = u.unsqueeze(-1)
    if u_max is not None:
        u = torch.clamp(u, -float(abs(u_max)), float(abs(u_max)))
    return x @ A.T + u @ B.T


def closed_loop_next(x: torch.Tensor, actor: ActorNetwork, dynamics: MotorDynamics) -> torch.Tensor:
    return dynamics(x, actor(x))


def _dataset_tensors(dataset: torch.utils.data.Dataset, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, u, v = zip(*(dataset[i] for i in range(len(dataset))))
    return torch.stack(x).to(device), torch.stack(u).to(device), torch.stack(v).to(device)


def _value_scale(v_star: torch.Tensor) -> float:
    return max(1.0, float(torch.quantile(torch.abs(v_star.detach().flatten()), 0.90).item()))


@torch.no_grad()
def _fit_linear_actor_gain(x: torch.Tensor, u_star: torch.Tensor) -> torch.Tensor:
    regularization = 1e-6 * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    gram = x.T @ x + regularization
    rhs = x.T @ u_star
    gain_t = torch.linalg.solve(gram, rhs)
    return gain_t.T


def _estimate_covering_radius(x: torch.Tensor) -> float:
    spacings = []
    for dim in range(x.shape[1]):
        values = torch.unique(torch.sort(x[:, dim]).values)
        if values.numel() > 1:
            diffs = values[1:] - values[:-1]
            positive = diffs[diffs > 1e-8]
            if positive.numel() > 0:
                spacings.append(float(positive.min().item()))
    if spacings:
        return float(np.sqrt(x.shape[1]) * max(spacings) / 2.0)
    if x.shape[0] < 2:
        return 0.0
    distances = torch.cdist(x, x)
    distances.fill_diagonal_(float("inf"))
    return float(0.5 * distances.min(dim=1).values.max().item())


def _estimate_empirical_lipschitz(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_pairs: int = 200_000,
) -> float:
    n = x.shape[0]
    if n < 2:
        return 0.0
    generator = torch.Generator(device=x.device).manual_seed(13)
    if n * (n - 1) // 2 <= max_pairs:
        distances = torch.cdist(x, x)
        value_diffs = torch.cdist(y, y, p=1)
        mask = distances > 1e-10
        return float((value_diffs[mask] / distances[mask]).max().item())
    idx_i = torch.randint(0, n, (max_pairs,), generator=generator, device=x.device)
    idx_j = torch.randint(0, n, (max_pairs,), generator=generator, device=x.device)
    mask = idx_i != idx_j
    dx = torch.linalg.norm(x[idx_i[mask]] - x[idx_j[mask]], dim=1).clamp_min(1e-10)
    dy = torch.abs(y[idx_i[mask]] - y[idx_j[mask]]).flatten()
    return float((dy / dx).max().item())


def _estimate_vhat_lipschitz(
    lyapunov: ICNNLyapunov,
    x: torch.Tensor,
    *,
    batch_size: int = 512,
) -> float:
    was_training = lyapunov.training
    lyapunov.eval()
    max_norm = 0.0
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size].detach().clone().requires_grad_(True)
        value = lyapunov.V_hat(batch).sum()
        grad = torch.autograd.grad(value, batch, create_graph=False)[0]
        max_norm = max(max_norm, float(torch.linalg.norm(grad, dim=1).max().item()))
    if was_training:
        lyapunov.train()
    return max_norm


def estimate_beta_bounds(
    lyapunov: ICNNLyapunov,
    x_val: torch.Tensor,
    v_star_val: torch.Tensor,
    *,
    kappa_lipschitz: float = 1.5,
    use_lipschitz_correction: bool = True,
    max_lipschitz_pairs: int = 200_000,
) -> BetaBounds:
    lyapunov.eval()
    with torch.no_grad():
        v_hat = lyapunov.V_hat(x_val)
        error = v_star_val - v_hat
        beta_minus_emp = max(0.0, float(error.max().item()))
        beta_plus_emp = max(0.0, float((-error).max().item()))

    L_V_emp = _estimate_empirical_lipschitz(x_val, v_star_val, max_pairs=max_lipschitz_pairs)
    L_V_cert = float(kappa_lipschitz * L_V_emp)
    L_Vhat_cert = _estimate_vhat_lipschitz(lyapunov, x_val)
    rho = _estimate_covering_radius(x_val)
    correction = (L_V_cert + L_Vhat_cert) * rho if use_lipschitz_correction else 0.0
    beta_minus_cert = beta_minus_emp + correction
    beta_plus_cert = beta_plus_emp + correction
    return BetaBounds(
        beta_minus_emp=beta_minus_emp,
        beta_plus_emp=beta_plus_emp,
        beta_minus_cert=beta_minus_cert,
        beta_plus_cert=beta_plus_cert,
        beta_symmetric_emp=max(beta_minus_emp, beta_plus_emp),
        beta_symmetric_cert=max(beta_minus_cert, beta_plus_cert),
        L_V_emp=L_V_emp,
        L_V_cert=L_V_cert,
        L_Vhat_cert=L_Vhat_cert,
        rho=rho,
    )


def lyapunov_losses(
    lyapunov: ICNNLyapunov,
    actor: ActorNetwork,
    dynamics: MotorDynamics,
    x: torch.Tensor,
    u_star: torch.Tensor,
    v_star: torch.Tensor,
    *,
    c_alpha: float,
    beta_minus: float,
    beta_plus: float,
    beta_symmetric: float,
    use_symmetric_beta: bool,
    lambda_value: float,
    lambda_actor: float,
    lambda_decrease: float,
    lambda_origin: float,
    lambda_rollout: float,
    lambda_positive: float,
    value_scale: float,
    rollout_steps: int,
    rollout_decay: float,
) -> dict[str, torch.Tensor]:
    v_hat = lyapunov.V_hat(x)
    u_hat = actor(x)
    x_next = dynamics(x, u_hat)
    delta_v_hat = lyapunov.V_hat(x_next) - v_hat
    alpha = c_alpha * torch.sum(x * x, dim=1, keepdim=True)
    if use_symmetric_beta:
        violation = delta_v_hat + 2.0 * beta_symmetric + alpha
    else:
        violation = delta_v_hat + beta_minus + beta_plus + alpha
    origin = torch.zeros((1, x.shape[1]), dtype=x.dtype, device=x.device)
    value_loss = nn.functional.mse_loss(v_hat / value_scale, v_star / value_scale)
    actor_loss = nn.functional.mse_loss(u_hat, u_star)
    decrease_loss = torch.relu(violation).pow(2).mean()
    origin_loss = (lyapunov.V_hat(origin) / value_scale).pow(2).mean()
    positive_loss = torch.relu(-v_hat / value_scale).pow(2).mean()
    rollout_loss = rollout_stability_loss(
        actor,
        dynamics,
        x,
        steps=rollout_steps,
        decay=rollout_decay,
    )
    total = (
        lambda_value * value_loss
        + lambda_actor * actor_loss
        + lambda_decrease * decrease_loss
        + lambda_origin * origin_loss
        + lambda_rollout * rollout_loss
        + lambda_positive * positive_loss
    )
    return {
        "total": total,
        "value": value_loss,
        "actor": actor_loss,
        "decrease": decrease_loss,
        "origin": origin_loss,
        "rollout": rollout_loss,
        "positive": positive_loss,
    }


def supervised_losses(
    lyapunov: ICNNLyapunov,
    actor: ActorNetwork,
    x: torch.Tensor,
    u_star: torch.Tensor,
    v_star: torch.Tensor,
    *,
    lambda_value: float,
    lambda_actor: float,
    lambda_origin: float,
    lambda_positive: float,
    value_scale: float,
) -> dict[str, torch.Tensor]:
    v_hat = lyapunov.V_hat(x)
    u_hat = actor(x)
    origin = torch.zeros((1, x.shape[1]), dtype=x.dtype, device=x.device)
    value_loss = nn.functional.mse_loss(v_hat / value_scale, v_star / value_scale)
    actor_loss = nn.functional.mse_loss(u_hat, u_star)
    origin_loss = (lyapunov.V_hat(origin) / value_scale).pow(2).mean()
    positive_loss = torch.relu(-v_hat / value_scale).pow(2).mean()
    total = (
        lambda_value * value_loss
        + lambda_actor * actor_loss
        + lambda_origin * origin_loss
        + lambda_positive * positive_loss
    )
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return {
        "total": total,
        "value": value_loss,
        "actor": actor_loss,
        "decrease": zero,
        "origin": origin_loss,
        "rollout": zero,
        "positive": positive_loss,
    }


def rollout_stability_loss(
    actor: ActorNetwork,
    dynamics: MotorDynamics,
    x: torch.Tensor,
    *,
    steps: int,
    decay: float,
) -> torch.Tensor:
    if steps <= 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    x_roll = x
    initial_norm = torch.linalg.norm(x, dim=1, keepdim=True).detach()
    previous_norm = initial_norm
    losses = []
    for _ in range(steps):
        x_roll = closed_loop_next(x_roll, actor, dynamics)
        current_norm = torch.linalg.norm(x_roll, dim=1, keepdim=True)
        losses.append(torch.relu(current_norm - decay * previous_norm).pow(2).mean())
        losses.append(0.25 * torch.relu(current_norm - initial_norm).pow(2).mean())
        previous_norm = current_norm
    return torch.stack(losses).mean()


def verify_stability(
    lyapunov: ICNNLyapunov,
    actor: ActorNetwork,
    dynamics: MotorDynamics,
    validation_states: torch.Tensor,
    *,
    v_star: torch.Tensor | None,
    beta_bounds: BetaBounds,
    c_alpha: float,
    use_symmetric_beta: bool,
    rollout_steps: int = 30,
    divergence_factor: float = 2.0,
    relative_norm_floor: float = 5e-2,
    absolute_norm_bound: float = 5.0,
) -> dict[str, float | list[float]]:
    lyapunov.eval()
    actor.eval()
    with torch.no_grad():
        x = validation_states
        x_next = closed_loop_next(x, actor, dynamics)
        delta_v_hat = lyapunov.V_hat(x_next) - lyapunov.V_hat(x)
        alpha = c_alpha * torch.sum(x * x, dim=1, keepdim=True)
        if use_symmetric_beta:
            robust_margin = delta_v_hat + 2.0 * beta_bounds.beta_symmetric_cert + alpha
        else:
            robust_margin = delta_v_hat + beta_bounds.beta_minus_cert + beta_bounds.beta_plus_cert + alpha
        worst_idx = int(torch.argmax(robust_margin).item())
        report: dict[str, float | list[float]] = {
            "max_delta_V_hat": float(delta_v_hat.max().item()),
            "max_robust_margin": float(robust_margin.max().item()),
            "percentage_satisfying_robust_condition": float((robust_margin <= 0.0).float().mean().item() * 100.0),
            "worst_state": x[worst_idx].detach().cpu().tolist(),
            "worst_violation": float(robust_margin[worst_idx].item()),
        }
        x_roll = x
        initial_norm_raw = torch.linalg.norm(x, dim=1)
        initial_norm = initial_norm_raw.clamp_min(relative_norm_floor)
        max_norm = initial_norm_raw.clone()
        for _ in range(max(rollout_steps, 0)):
            x_roll = closed_loop_next(x_roll, actor, dynamics)
            max_norm = torch.maximum(max_norm, torch.linalg.norm(x_roll, dim=1))
        growth = max_norm / initial_norm
        report.update(
            {
                "max_rollout_norm_growth": float(growth.max().item()),
                "percentage_rollouts_not_diverging": float((growth <= divergence_factor).float().mean().item() * 100.0),
                "max_rollout_abs_norm": float(max_norm.max().item()),
                "percentage_rollouts_abs_bounded": float((max_norm <= absolute_norm_bound).float().mean().item() * 100.0),
                "relative_norm_floor": float(relative_norm_floor),
            }
        )
        if v_star is not None:
            distances = torch.cdist(x_next, x)
            nn_idx = torch.argmin(distances, dim=1)
            v_next_star = v_star[nn_idx]
            delta_v_star = v_next_star - v_star
            true_margin = delta_v_star + alpha
            report.update(
                {
                    "max_delta_V_star": float(delta_v_star.max().item()),
                    "percentage_satisfying_true_decrease": float((true_margin <= 0.0).float().mean().item() * 100.0),
                    "max_true_value_nn_distance": float(distances.min(dim=1).values.max().item()),
                }
            )
    return report


def simulate_actor_trajectory(
    actor: ActorNetwork,
    dynamics: MotorDynamics,
    x0: np.ndarray,
    *,
    nsim: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    x = torch.as_tensor(x0, dtype=torch.float32, device=device).reshape(1, -1)
    x_hist = [x.detach().cpu().numpy().reshape(-1)]
    u_hist = []
    with torch.no_grad():
        for _ in range(nsim):
            u = actor(x)
            x = dynamics(x, u)
            u_hist.append(u.detach().cpu().numpy().reshape(-1))
            x_hist.append(x.detach().cpu().numpy().reshape(-1))
    return {"x": np.asarray(x_hist), "u": np.asarray(u_hist)}


def _select_trajectory_seeds(x: torch.Tensor, *, n_seeds: int = 6) -> np.ndarray:
    x_cpu = x.detach().cpu()
    norms = torch.linalg.norm(x_cpu, dim=1)
    if x_cpu.shape[0] <= n_seeds:
        return x_cpu.numpy()
    candidate_ids = [
        int(torch.argmin(x_cpu[:, 0]).item()),
        int(torch.argmax(x_cpu[:, 0]).item()),
        int(torch.argmin(x_cpu[:, 1]).item()),
        int(torch.argmax(x_cpu[:, 1]).item()),
        int(torch.argmax(norms).item()),
        int(torch.argmin(norms).item()),
    ]
    unique_ids = []
    for idx in candidate_ids:
        if idx not in unique_ids:
            unique_ids.append(idx)
    if len(unique_ids) < n_seeds:
        quantiles = torch.linspace(0.15, 0.95, n_seeds)
        sorted_ids = torch.argsort(norms)
        for q in quantiles:
            idx = int(sorted_ids[min(int(q.item() * (len(sorted_ids) - 1)), len(sorted_ids) - 1)].item())
            if idx not in unique_ids:
                unique_ids.append(idx)
            if len(unique_ids) >= n_seeds:
                break
    return x_cpu[unique_ids[:n_seeds]].numpy()


def _ordered_index(x: torch.Tensor) -> np.ndarray:
    x_np = x.detach().cpu().numpy()
    return np.lexsort((x_np[:, 1], x_np[:, 0]))


def _save_history_plot(history: TrainingHistory, output_path: Path) -> None:
    if not history.epoch:
        return
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axs[0].semilogy(history.epoch, history.train_value, label="train value")
    axs[0].semilogy(history.epoch, history.val_value, label="val value")
    axs[0].semilogy(history.epoch, history.train_actor, label="train actor")
    axs[0].semilogy(history.epoch, history.val_actor, label="val actor")
    axs[0].set_ylabel("supervised loss")
    axs[0].grid(True)
    axs[0].legend()
    axs[1].semilogy(history.epoch, np.maximum(history.train_decrease, 1e-16), label="train decrease")
    axs[1].semilogy(history.epoch, np.maximum(history.val_decrease, 1e-16), label="val decrease")
    axs[1].semilogy(history.epoch, np.maximum(history.train_rollout, 1e-16), label="train rollout")
    axs[1].semilogy(history.epoch, np.maximum(history.val_rollout, 1e-16), label="val rollout")
    axs[1].plot(history.epoch, history.beta_minus_cert, label="beta_minus_cert")
    axs[1].plot(history.epoch, history.beta_plus_cert, label="beta_plus_cert")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("robust terms")
    axs[1].grid(True)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(output_path / "00_training_history.png", dpi=200)
    plt.close(fig)


def _rollout_norm_matrix(
    actor: ActorNetwork,
    dynamics: MotorDynamics,
    x: torch.Tensor,
    *,
    nsim: int,
) -> np.ndarray:
    norms = []
    x_roll = x
    with torch.no_grad():
        norms.append(torch.linalg.norm(x_roll, dim=1).detach().cpu().numpy())
        for _ in range(nsim):
            x_roll = closed_loop_next(x_roll, actor, dynamics)
            norms.append(torch.linalg.norm(x_roll, dim=1).detach().cpu().numpy())
    return np.asarray(norms)


def plot_results(
    lyapunov: ICNNLyapunov,
    actor: ActorNetwork,
    dynamics: MotorDynamics,
    x_val: torch.Tensor,
    u_val: torch.Tensor,
    v_star_val: torch.Tensor,
    beta_bounds: BetaBounds,
    *,
    c_alpha: float,
    use_symmetric_beta: bool,
    output_dir: str | Path,
    nsim: int = 50,
    history: TrainingHistory | None = None,
) -> None:
    output_path = _resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if history is not None:
        _save_history_plot(history, output_path)
    lyapunov.eval()
    actor.eval()
    with torch.no_grad():
        v_hat = lyapunov.V_hat(x_val)
        error = v_star_val - v_hat
        x_next = closed_loop_next(x_val, actor, dynamics)
        delta_v_hat = lyapunov.V_hat(x_next) - v_hat
        alpha = c_alpha * torch.sum(x_val * x_val, dim=1, keepdim=True)
        if use_symmetric_beta:
            robust_violation = delta_v_hat + 2.0 * beta_bounds.beta_symmetric_cert + alpha
            tube = beta_bounds.beta_symmetric_cert
        else:
            robust_violation = delta_v_hat + beta_bounds.beta_minus_cert + beta_bounds.beta_plus_cert + alpha
            tube = max(beta_bounds.beta_minus_cert, beta_bounds.beta_plus_cert)

    arrays = {
        "v_star": v_star_val.detach().cpu().numpy().reshape(-1),
        "v_hat": v_hat.detach().cpu().numpy().reshape(-1),
        "error": error.detach().cpu().numpy().reshape(-1),
        "delta": delta_v_hat.detach().cpu().numpy().reshape(-1),
        "violation": robust_violation.detach().cpu().numpy().reshape(-1),
    }
    x_np = x_val.detach().cpu().numpy()
    u_np = u_val.detach().cpu().numpy().reshape(-1)
    with torch.no_grad():
        u_hat_np = actor(x_val).detach().cpu().numpy().reshape(-1)
    order = _ordered_index(x_val)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(arrays["v_star"], arrays["v_hat"], s=12, alpha=0.75)
    lo = min(arrays["v_star"].min(), arrays["v_hat"].min())
    hi = max(arrays["v_star"].max(), arrays["v_hat"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("V_star")
    ax.set_ylabel("V_hat")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path / "01_v_star_vs_v_hat.png", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axs[0].plot(arrays["v_star"][order], label="V_star", linewidth=1.5)
    axs[0].plot(arrays["v_hat"][order], label="V_hat", linewidth=1.5)
    axs[0].set_ylabel("value")
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(arrays["error"][order], linewidth=1)
    axs[1].axhline(beta_bounds.beta_minus_cert, color="tab:red", linestyle="--", label="beta_minus_cert")
    axs[1].axhline(-beta_bounds.beta_plus_cert, color="tab:blue", linestyle="--", label="-beta_plus_cert")
    axs[1].set_xlabel("validation states ordered by x1, x2")
    axs[1].set_ylabel("V_star - V_hat")
    axs[1].grid(True)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(output_path / "02_value_error.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(arrays["v_star"], arrays["v_hat"], s=12, alpha=0.7)
    ax.plot([lo, hi], [lo + tube, hi + tube], "tab:blue", linestyle="--", linewidth=1)
    ax.plot([lo, hi], [lo - tube, hi - tube], "tab:red", linestyle="--", linewidth=1)
    ax.set_xlabel("V_star")
    ax.set_ylabel("V_hat certified tube")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path / "03_beta_certified_tube.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(arrays["delta"][order], linewidth=1)
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xlabel("validation states ordered by x1, x2")
    ax.set_ylabel("Delta V_hat")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path / "04_delta_v_hat.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(arrays["violation"][order], linewidth=1)
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xlabel("validation states ordered by x1, x2")
    ax.set_ylabel("Robust violation")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path / "05_robust_violation.png", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    scatter_specs = [
        ("V_star", arrays["v_star"]),
        ("V_hat", arrays["v_hat"]),
        ("error", arrays["error"]),
        ("robust violation", arrays["violation"]),
    ]
    for ax, (title, values) in zip(axs.flat, scatter_specs):
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], c=values, s=20, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True)
        fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path / "06_state_space_diagnostics.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u_np, u_hat_np, s=14, alpha=0.75)
    lo_u = min(u_np.min(), u_hat_np.min())
    hi_u = max(u_np.max(), u_hat_np.max())
    ax.plot([lo_u, hi_u], [lo_u, hi_u], "k--", linewidth=1)
    ax.set_xlabel("u_star")
    ax.set_ylabel("actor u")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path / "07_actor_fit.png", dpi=200)
    plt.close(fig)

    seeds = _select_trajectory_seeds(x_val, n_seeds=6)
    trajectories = [simulate_actor_trajectory(actor, dynamics, x0, nsim=nsim, device=x_val.device) for x0 in seeds]

    fig, ax = plt.subplots(figsize=(6, 5))
    for traj in trajectories:
        ax.plot(traj["x"][:, 0], traj["x"][:, 1], marker="o", markersize=2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path / "08_closed_loop_trajectories.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in trajectories:
        ax.plot(np.linalg.norm(traj["x"], axis=1))
    ax.set_xlabel("k")
    ax.set_ylabel("||x||")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path / "09_state_norm_decay.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in trajectories:
        ax.step(np.arange(traj["u"].shape[0]), traj["u"][:, 0], where="post")
    ax.set_xlabel("k")
    ax.set_ylabel("u")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path / "10_control_input.png", dpi=200)
    plt.close(fig)

    norm_matrix = _rollout_norm_matrix(actor, dynamics, x_val, nsim=nsim)
    quantiles = np.quantile(norm_matrix, [0.0, 0.5, 0.9, 1.0], axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(norm_matrix.shape[0])
    ax.plot(t, quantiles[1], label="median")
    ax.plot(t, quantiles[2], label="90%")
    ax.plot(t, quantiles[3], label="max")
    ax.fill_between(t, quantiles[0], quantiles[3], alpha=0.15, label="min-max")
    ax.set_xlabel("k")
    ax.set_ylabel("validation rollout ||x||")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "11_all_validation_rollout_norms.png", dpi=200)
    plt.close(fig)


def train_value_actor(args: argparse.Namespace) -> tuple[ICNNLyapunov, ActorNetwork, BetaBounds, dict[str, float | list[float]]]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = make_value_actor_dataloaders(args.dataset, batch_size=args.batch_size, seed=args.seed)

    lyapunov = ICNNLyapunov(
        input_dim=bundle.x.shape[1],
        hidden_sizes=tuple(args.value_hidden_sizes),
        activation=args.value_activation,
        eps_positive_definite=args.eps_positive_definite,
    ).to(device)
    lyapunov.set_input_normalization(bundle.input_mean.to(device), bundle.input_std.to(device))
    x_train_full, u_train_full, v_train_full = _dataset_tensors(bundle.train_dataset, device=device)
    value_scale = _value_scale(v_train_full)
    actor = ActorNetwork(
        state_dim=bundle.x.shape[1],
        control_dim=bundle.u_star.shape[1],
        hidden_sizes=tuple(args.actor_hidden_sizes),
        activation=args.actor_activation,
        u_max=args.u_max,
        residual_scale=args.actor_residual_scale,
    ).to(device)
    actor.set_linear_skip(_fit_linear_actor_gain(x_train_full, u_train_full))
    dynamics = MotorDynamics(vertex_id=args.vertex_id, u_max=args.u_max, device=device).to(device)

    optimizer = torch.optim.Adam(
        list(lyapunov.parameters()) + list(actor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    x_val, u_val, v_val = _dataset_tensors(bundle.val_dataset, device=device)
    beta_bounds = estimate_beta_bounds(
        lyapunov,
        x_val,
        v_val,
        kappa_lipschitz=args.kappa_lipschitz,
        use_lipschitz_correction=args.use_lipschitz_correction,
        max_lipschitz_pairs=args.max_lipschitz_pairs,
    )

    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    history = TrainingHistory.empty()
    total_epochs = int(args.pretrain_epochs + args.epochs)
    for epoch in range(1, total_epochs + 1):
        is_pretrain = epoch <= args.pretrain_epochs
        if epoch == args.pretrain_epochs + 1:
            patience_left = args.patience
        lyapunov.train()
        actor.train()
        running = {"total": 0.0, "value": 0.0, "actor": 0.0, "decrease": 0.0, "rollout": 0.0}
        n_train = 0
        for x, u_star, v_star in bundle.train_loader:
            x = x.to(device)
            u_star = u_star.to(device)
            v_star = v_star.to(device)
            optimizer.zero_grad(set_to_none=True)
            if is_pretrain:
                losses = supervised_losses(
                    lyapunov,
                    actor,
                    x,
                    u_star,
                    v_star,
                    lambda_value=args.lambda_value,
                    lambda_actor=args.lambda_actor,
                    lambda_origin=args.lambda_origin,
                    lambda_positive=args.lambda_positive,
                    value_scale=value_scale,
                )
            else:
                losses = lyapunov_losses(
                    lyapunov,
                    actor,
                    dynamics,
                    x,
                    u_star,
                    v_star,
                    c_alpha=args.c_alpha,
                    beta_minus=beta_bounds.beta_minus_cert if args.beta_in_training_loss else 0.0,
                    beta_plus=beta_bounds.beta_plus_cert if args.beta_in_training_loss else 0.0,
                    beta_symmetric=beta_bounds.beta_symmetric_cert if args.beta_in_training_loss else 0.0,
                    use_symmetric_beta=args.use_symmetric_beta,
                    lambda_value=args.lambda_value,
                    lambda_actor=args.lambda_actor,
                    lambda_decrease=args.lambda_decrease,
                    lambda_origin=args.lambda_origin,
                    lambda_rollout=args.lambda_rollout,
                    lambda_positive=args.lambda_positive,
                    value_scale=value_scale,
                    rollout_steps=args.rollout_steps,
                    rollout_decay=args.rollout_decay,
                )
            losses["total"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(lyapunov.parameters()) + list(actor.parameters()),
                    max_norm=args.grad_clip,
                )
            optimizer.step()
            lyapunov.project_nonnegative_weights()
            batch_n = x.shape[0]
            n_train += batch_n
            for key in running:
                running[key] += float(losses[key].item()) * batch_n

        if epoch == 1 or epoch == args.pretrain_epochs + 1 or epoch % args.beta_update_every == 0:
            beta_bounds = estimate_beta_bounds(
                lyapunov,
                x_val,
                v_val,
                kappa_lipschitz=args.kappa_lipschitz,
                use_lipschitz_correction=args.use_lipschitz_correction,
                max_lipschitz_pairs=args.max_lipschitz_pairs,
            )

        lyapunov.eval()
        actor.eval()
        with torch.no_grad():
            if is_pretrain:
                val_losses = supervised_losses(
                    lyapunov,
                    actor,
                    x_val,
                    u_val,
                    v_val,
                    lambda_value=args.lambda_value,
                    lambda_actor=args.lambda_actor,
                    lambda_origin=args.lambda_origin,
                    lambda_positive=args.lambda_positive,
                    value_scale=value_scale,
                )
            else:
                val_losses = lyapunov_losses(
                    lyapunov,
                    actor,
                    dynamics,
                    x_val,
                    u_val,
                    v_val,
                    c_alpha=args.c_alpha,
                    beta_minus=beta_bounds.beta_minus_cert if args.beta_in_training_loss else 0.0,
                    beta_plus=beta_bounds.beta_plus_cert if args.beta_in_training_loss else 0.0,
                    beta_symmetric=beta_bounds.beta_symmetric_cert if args.beta_in_training_loss else 0.0,
                    use_symmetric_beta=args.use_symmetric_beta,
                    lambda_value=args.lambda_value,
                    lambda_actor=args.lambda_actor,
                    lambda_decrease=args.lambda_decrease,
                    lambda_origin=args.lambda_origin,
                    lambda_rollout=args.lambda_rollout,
                    lambda_positive=args.lambda_positive,
                    value_scale=value_scale,
                    rollout_steps=args.rollout_steps,
                    rollout_decay=args.rollout_decay,
                )
        val_total = float(val_losses["total"].item())
        history.epoch.append(epoch)
        history.train_total.append(running["total"] / n_train)
        history.train_value.append(running["value"] / n_train)
        history.train_actor.append(running["actor"] / n_train)
        history.train_decrease.append(running["decrease"] / n_train)
        history.train_rollout.append(running["rollout"] / n_train)
        history.val_total.append(val_total)
        history.val_value.append(float(val_losses["value"].item()))
        history.val_actor.append(float(val_losses["actor"].item()))
        history.val_decrease.append(float(val_losses["decrease"].item()))
        history.val_rollout.append(float(val_losses["rollout"].item()))
        history.beta_minus_cert.append(beta_bounds.beta_minus_cert)
        history.beta_plus_cert.append(beta_bounds.beta_plus_cert)
        if val_total < best_val:
            best_val = val_total
            best_state = {
                "lyapunov": {key: value.detach().cpu().clone() for key, value in lyapunov.state_dict().items()},
                "actor": {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()},
            }
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch == 1 or epoch % args.print_every == 0:
            phase = "pretrain" if is_pretrain else "robust"
            print(
                f"epoch {epoch:04d} ({phase}) | train {running['total'] / n_train:.6e} | "
                f"val {val_total:.6e} | value {float(val_losses['value'].item()):.6e} | "
                f"actor {float(val_losses['actor'].item()):.6e} | decrease {float(val_losses['decrease'].item()):.6e} | "
                f"rollout {float(val_losses['rollout'].item()):.6e} | "
                f"beta- {beta_bounds.beta_minus_cert:.3e} beta+ {beta_bounds.beta_plus_cert:.3e}"
            )
        if patience_left <= 0 and not is_pretrain:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        lyapunov.load_state_dict(best_state["lyapunov"])
        actor.load_state_dict(best_state["actor"])
    lyapunov.project_nonnegative_weights()

    x_test, u_test, v_test = _dataset_tensors(bundle.test_dataset, device=device)
    beta_bounds = estimate_beta_bounds(
        lyapunov,
        x_val,
        v_val,
        kappa_lipschitz=args.kappa_lipschitz,
        use_lipschitz_correction=args.use_lipschitz_correction,
        max_lipschitz_pairs=args.max_lipschitz_pairs,
    )
    report = verify_stability(
        lyapunov,
        actor,
        dynamics,
        x_test,
        v_star=v_test,
        beta_bounds=beta_bounds,
        c_alpha=args.c_alpha,
        use_symmetric_beta=args.use_symmetric_beta,
        rollout_steps=max(args.plot_nsim, args.rollout_steps),
    )

    print("Beta/stability report")
    for key, value in beta_bounds.__dict__.items():
        print(f"  {key}: {value:.6e}")
    for key, value in report.items():
        print(f"  {key}: {value}")

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lyapunov_state_dict": lyapunov.state_dict(),
            "actor_state_dict": actor.state_dict(),
            "state_dim": bundle.x.shape[1],
            "control_dim": bundle.u_star.shape[1],
            "value_hidden_sizes": tuple(args.value_hidden_sizes),
            "actor_hidden_sizes": tuple(args.actor_hidden_sizes),
            "value_activation": args.value_activation,
            "actor_activation": args.actor_activation,
            "eps_positive_definite": args.eps_positive_definite,
            "u_max": args.u_max,
            "vertex_id": args.vertex_id,
            "input_mean": bundle.input_mean,
            "input_std": bundle.input_std,
            "value_scale": value_scale,
            "value_column": bundle.value_column,
            "beta_bounds": beta_bounds.__dict__,
            "stability_report": report,
            "training_history": history.__dict__,
        },
        output_path,
    )
    print(f"Saved value/actor checkpoint to {output_path}")

    if args.plot_dir:
        plot_results(
            lyapunov,
            actor,
            dynamics,
            x_val,
            u_val,
            v_val,
            beta_bounds,
            c_alpha=args.c_alpha,
            use_symmetric_beta=args.use_symmetric_beta,
            output_dir=args.plot_dir,
            nsim=args.plot_nsim,
            history=history,
        )
        print(f"Saved plots to {_resolve_path(args.plot_dir)}")
    return lyapunov, actor, beta_bounds, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ICNN Lyapunov and actor networks for the motor example.")
    parser.add_argument("--dataset", default="icnn_lyapunov/simulation_data_theta_small.csv")
    parser.add_argument("--output", default="icnn_lyapunov/value_actor.pth")
    parser.add_argument("--plot-dir", default="icnn_lyapunov/value_actor_plots")
    parser.add_argument("--pretrain-epochs", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--value-hidden-sizes", type=int, nargs="+", default=[64, 64, 32])
    parser.add_argument("--actor-hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--value-activation", choices=["relu", "softplus"], default="softplus")
    parser.add_argument("--actor-activation", choices=["relu", "tanh", "softplus"], default="tanh")
    parser.add_argument("--actor-residual-scale", type=float, default=0.25)
    parser.add_argument("--vertex-id", type=int, default=1)
    parser.add_argument("--u-max", type=float, default=2.0)
    parser.add_argument("--c-alpha", type=float, default=1e-4)
    parser.add_argument("--eps-positive-definite", type=float, default=1e-4)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--lambda-actor", type=float, default=5.0)
    parser.add_argument("--lambda-decrease", type=float, default=0.1)
    parser.add_argument("--lambda-origin", type=float, default=1.0)
    parser.add_argument("--lambda-positive", type=float, default=10.0)
    parser.add_argument("--lambda-rollout", type=float, default=20.0)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--rollout-decay", type=float, default=1.0)
    parser.add_argument("--kappa-lipschitz", type=float, default=1.5)
    parser.add_argument("--use-symmetric-beta", action="store_true")
    parser.add_argument("--use-lipschitz-correction", action="store_true")
    parser.add_argument("--beta-in-training-loss", action="store_true")
    parser.add_argument("--max-lipschitz-pairs", type=int, default=200000)
    parser.add_argument("--beta-update-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--plot-nsim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train_value_actor(parse_args())
