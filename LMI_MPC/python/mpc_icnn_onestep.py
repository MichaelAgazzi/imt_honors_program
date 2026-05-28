import time
from pathlib import Path

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, TensorDataset, random_split

from motor import A_list, B_list


# ==========================================================
# CONFIGURATION
# ==========================================================
DT = 0.05
N = 100
N_SIM = 120
VERTEX_ID = 1
U_BOUND = 2.0
X0 = np.array([[0.7], [0.7]])
MULTI_X1 = np.linspace(-0.8, 0.8, 10)
MULTI_X2 = np.linspace(-0.9, 0.9, 10)
MULTI_X0 = [
    np.array([[x1], [x2]])
    for x1 in MULTI_X1
    for x2 in MULTI_X2
]
COMPARISON_X0 = [
    np.array([[0.7], [0.7]]),
    np.array([[0.7], [-0.5]]),
    np.array([[-0.7], [0.7]]),
    np.array([[-0.7], [-0.5]]),
    np.array([[0.3], [0.9]]),
    np.array([[-0.3], [-0.9]]),
]

Q = np.diag([1.0, 0.1])
R = np.array([[2e-3]])
QF = 10.0 * Q

N_DATA = 2500
TRAIN_EPOCHS = 1800
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
RANDOM_SEED = 7
LOCAL_DATA_FRACTION = 0.45
FORCE_REBUILD_DATASET = False
FORCE_RETRAIN_ICNN = False

STATE_LOW = np.array([-1.2, -1.0])
STATE_HIGH = np.array([1.2, 1.0])

OUT_DIR = Path("figure")
OUT_DIR.mkdir(exist_ok=True)
DATASET_PATH = Path(f"mpc_tail_cost_dataset_N{N}.csv")
MODEL_PATH = Path(f"mpc_tail_cost_icnn_N{N}.pth")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ==========================================================
# NOMINAL MODEL
# ==========================================================
# The project motor model is discrete-time with nominal vertices in motor.py.
# We use the same nominal vertex used in existing simulations.
A_NOM = A_list[VERTEX_ID].astype(float)
B_NOM = B_list[VERTEX_ID].astype(float)


def step_nominal(x, u):
    return A_NOM @ x + B_NOM * float(u)


def stage_cost(x, u):
    x = np.asarray(x, dtype=float).reshape(2, 1)
    u = float(u)
    return float((x.T @ Q @ x).item() + u * R[0, 0] * u)


def terminal_cost(x):
    x = np.asarray(x, dtype=float).reshape(2, 1)
    return float((x.T @ QF @ x).item())


# ==========================================================
# FINITE-HORIZON MPC SOLVER
# ==========================================================
class LinearMPC:
    def __init__(self, horizon):
        self.horizon = horizon
        self.x0 = cp.Parameter(2)
        self.x = cp.Variable((2, horizon + 1))
        self.u = cp.Variable(horizon)

        cost = 0
        constraints = [self.x[:, 0] == self.x0]
        for k in range(horizon):
            cost += cp.quad_form(self.x[:, k], Q) + R[0, 0] * cp.square(self.u[k])
            constraints += [self.x[:, k + 1] == A_NOM @ self.x[:, k] + B_NOM[:, 0] * self.u[k]]
            constraints += [self.u[k] <= U_BOUND, self.u[k] >= -U_BOUND]
        cost += cp.quad_form(self.x[:, horizon], QF)

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, x0):
        self.x0.value = np.asarray(x0, dtype=float).reshape(2)
        t0 = time.perf_counter()
        try:
            self.problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        except Exception:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        solve_time = time.perf_counter() - t0

        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC infeasible or failed: {self.problem.status}")

        return float(self.u.value[0]), float(self.problem.value), solve_time


def rollout_nominal_mpc(x0):
    controller = LinearMPC(N)
    x = x0.copy()
    xs = [x.reshape(2)]
    us = []
    costs = []
    solve_times = []

    for _ in range(N_SIM):
        u, value, solve_time = controller.solve(x)
        x = step_nominal(x, u)
        xs.append(x.reshape(2))
        us.append(u)
        costs.append(value)
        solve_times.append(solve_time)

    return {
        "x": np.array(xs),
        "u": np.array(us),
        "value": np.array(costs),
        "solve_time": np.array(solve_times),
    }


# ==========================================================
# DATASET: TAIL COST V*(x) FROM STEP 1 TO N
# ==========================================================
def build_tail_cost_dataset(force_rebuild=False):
    if DATASET_PATH.exists() and not force_rebuild:
        df = pd.read_csv(DATASET_PATH)
        if len(df) >= N_DATA:
            return df[["x1", "x2"]].to_numpy(), df["tail_cost"].to_numpy()
        print(f"Existing dataset has {len(df)} samples, rebuilding with {N_DATA} samples.")

    tail_solver = LinearMPC(N - 1)
    n_local = int(LOCAL_DATA_FRACTION * N_DATA)
    n_global = N_DATA - n_local
    xs_global = np.random.uniform(STATE_LOW, STATE_HIGH, size=(n_global, 2))
    xs_local = np.random.normal(loc=0.0, scale=np.array([0.35, 0.35]), size=(n_local, 2))
    xs_local = np.clip(xs_local, STATE_LOW, STATE_HIGH)
    xs = np.vstack([xs_global, xs_local])
    np.random.shuffle(xs)
    values = []
    kept_xs = []

    for i, x in enumerate(xs, start=1):
        try:
            _, value, _ = tail_solver.solve(x.reshape(2, 1))
        except RuntimeError:
            continue
        kept_xs.append(x)
        values.append(value)
        if i % 100 == 0:
            print(f"Dataset solved: {i}/{N_DATA}")

    kept_xs = np.array(kept_xs)
    values = np.array(values)
    df = pd.DataFrame({"x1": kept_xs[:, 0], "x2": kept_xs[:, 1], "tail_cost": values})
    df.to_csv(DATASET_PATH, index=False)
    return kept_xs, values


# ==========================================================
# ICNN VALUE APPROXIMATOR
# ==========================================================
class TailCostICNN(nn.Module):
    def __init__(self, input_dim=2, hidden_sizes=(128, 128, 64)):
        super().__init__()
        self.quad_raw = nn.Parameter(torch.ones(input_dim))
        self.quad_factor = nn.Parameter(torch.eye(input_dim))
        self.hidden_sizes = list(hidden_sizes)
        self.Wz = nn.ModuleList()
        self.Wx = nn.ModuleList()

        for i, hidden in enumerate(self.hidden_sizes):
            if i == 0:
                self.Wz.append(nn.Linear(hidden, hidden, bias=False))
            else:
                self.Wz.append(nn.Linear(self.hidden_sizes[i - 1], hidden, bias=False))
            self.Wx.append(nn.Linear(input_dim, hidden))

        self.Wz_out = nn.Linear(self.hidden_sizes[-1], 1, bias=False)
        self.Wx_out = nn.Linear(input_dim, 1)

    def forward(self, x):
        z = torch.zeros(x.shape[0], self.hidden_sizes[0], device=x.device)
        for Wz, Wx in zip(self.Wz, self.Wx):
            z = F.softplus(Wz(z) + Wx(x))
        diag_quad = torch.sum(F.softplus(self.quad_raw) * x**2, dim=1, keepdim=True)
        full_quad = torch.sum((x @ self.quad_factor.T) ** 2, dim=1, keepdim=True)
        quad = diag_quad + full_quad
        return quad + F.softplus(self.Wz_out(z) + self.Wx_out(x))

    def project_nonnegative(self):
        with torch.no_grad():
            for layer in self.Wz:
                layer.weight.clamp_(min=0.0)
            self.Wz_out.weight.clamp_(min=0.0)


def train_icnn(xs, values):
    x_mean = xs.mean(axis=0)
    x_std = xs.std(axis=0) + 1e-8
    y_scale = np.percentile(values, 95) + 1e-8

    x_scaled = (xs - x_mean) / x_std
    y_scaled = values / y_scale

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled[:, None], dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = TailCostICNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    def weighted_loss(pred, target):
        weights = 1.0 / (0.05 + target)
        mse = torch.mean((pred - target) ** 2)
        local = torch.mean(weights * (pred - target) ** 2)
        return mse + 0.1 * local
    train_hist = []
    val_hist = []

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = weighted_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            model.project_nonnegative()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= train_size
        train_hist.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += weighted_loss(model(xb), yb).item() * xb.shape[0]
        val_loss /= val_size
        val_hist.append(val_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | train {train_loss:.3e} | val {val_loss:.3e}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": 0.0,
            "y_std": y_scale,
            "target_mode": "positive_scaled_cost_fullquad_icnn_v4",
            "dataset_size": len(xs),
            "train_epochs": TRAIN_EPOCHS,
            "train_loss": train_hist,
            "val_loss": val_hist,
        },
        MODEL_PATH,
    )

    plot_training(train_hist, val_hist)
    plot_value_fit(model, xs, values, x_mean, x_std, 0.0, y_scale)
    return model, x_mean, x_std, 0.0, y_scale


def load_or_train_icnn(xs, values, force_train=False):
    if MODEL_PATH.exists() and not force_train:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        checkpoint_is_current = (
            checkpoint.get("target_mode") == "positive_scaled_cost_fullquad_icnn_v4"
            and checkpoint.get("dataset_size", 0) >= len(xs)
            and checkpoint.get("train_epochs", 0) >= TRAIN_EPOCHS
        )
        if not checkpoint_is_current:
            return train_icnn(xs, values)
        model = TailCostICNN()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        plot_training(checkpoint["train_loss"], checkpoint["val_loss"])
        plot_value_fit(
            model,
            xs,
            values,
            checkpoint["x_mean"],
            checkpoint["x_std"],
            checkpoint["y_mean"],
            checkpoint["y_std"],
        )
        return (
            model,
            checkpoint["x_mean"],
            checkpoint["x_std"],
            checkpoint["y_mean"],
            checkpoint["y_std"],
        )
    return train_icnn(xs, values)


def make_value_predictor(model, x_mean, x_std, y_mean, y_std):
    model.eval()

    def predict(x):
        x = np.asarray(x, dtype=float).reshape(1, 2)
        x_scaled = (x - x_mean) / x_std
        with torch.no_grad():
            y_scaled = model(torch.tensor(x_scaled, dtype=torch.float32)).item()
        return float(y_scaled * y_std + y_mean)

    return predict


# ==========================================================
# ONE-STEP MPC WITH ICNN TAIL COST
# ==========================================================
def one_step_icnn_control(x, value_predictor):
    x_col = np.asarray(x, dtype=float).reshape(2, 1)

    def objective(u):
        x_next = step_nominal(x_col, u)
        return stage_cost(x_col, u) + value_predictor(x_next.reshape(2))

    t0 = time.perf_counter()
    result = minimize_scalar(
        objective,
        bounds=(-U_BOUND, U_BOUND),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 80},
    )
    solve_time = time.perf_counter() - t0
    u = float(np.clip(result.x, -U_BOUND, U_BOUND))
    return u, float(result.fun), solve_time


def rollout_one_step_icnn(x0, value_predictor):
    x = x0.copy()
    xs = [x.reshape(2)]
    us = []
    costs = []
    solve_times = []

    for _ in range(N_SIM):
        u, value, solve_time = one_step_icnn_control(x, value_predictor)
        x = step_nominal(x, u)
        xs.append(x.reshape(2))
        us.append(u)
        costs.append(value)
        solve_times.append(solve_time)

    return {
        "x": np.array(xs),
        "u": np.array(us),
        "value": np.array(costs),
        "solve_time": np.array(solve_times),
    }


# ==========================================================
# PLOTS AND REPORTS
# ==========================================================
def closed_loop_cost(result):
    x = result["x"]
    u = result["u"]
    total = 0.0
    for k in range(len(u)):
        total += stage_cost(x[k].reshape(2, 1), u[k])
    return total


def plot_training(train_hist, val_hist):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(train_hist, label="train")
    axs[0].plot(val_hist, label="validation")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("weighted MSE")
    axs[0].set_title("ICNN training loss, log scale")
    axs[0].grid(True, which="both")
    axs[0].legend()

    axs[1].plot(train_hist, label="train")
    axs[1].plot(val_hist, label="validation")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("weighted MSE")
    axs[1].set_title("ICNN training loss, linear scale")
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_icnn_training.png", dpi=250)
    plt.close()


def plot_value_fit(model, xs, values, x_mean, x_std, y_mean, y_std):
    n_plot = min(1200, len(xs))
    idx = np.linspace(0, len(xs) - 1, n_plot).astype(int)
    xs_plot = xs[idx]
    y_true = values[idx]

    x_scaled = (xs_plot - x_mean) / x_std
    with torch.no_grad():
        y_scaled = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy().reshape(-1)
    y_pred = y_scaled * y_std + y_mean
    abs_error = np.abs(y_pred - y_true)

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)
    plt.xlabel("MPC tail cost target")
    plt.ylabel("ICNN prediction")
    plt.title("Tail-cost fit")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(np.linalg.norm(xs_plot, axis=1), abs_error, s=10, alpha=0.6)
    plt.yscale("log")
    plt.xlabel("||x||")
    plt.ylabel("|prediction error|")
    plt.title("Fit error vs state norm")
    plt.grid(True, which="both")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_icnn_value_fit.png", dpi=250)
    plt.close()


def plot_comparison(nominal, icnn):
    t = np.arange(N_SIM + 1) * DT
    tu = np.arange(N_SIM) * DT

    plt.figure(figsize=(9, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, nominal["x"][:, 0], label="MPC N=20")
    plt.plot(t, icnn["x"][:, 0], "--", label="one-step ICNN")
    plt.ylabel("x1")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, nominal["x"][:, 1], label="MPC N=20")
    plt.plot(t, icnn["x"][:, 1], "--", label="one-step ICNN")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.step(tu, nominal["u"], where="post", label="MPC N=20")
    plt.step(tu, icnn["u"], where="post", linestyle="--", label="one-step ICNN")
    plt.axhline(U_BOUND, color="k", linestyle=":")
    plt.axhline(-U_BOUND, color="k", linestyle=":")
    plt.xlabel("time [s]")
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_nominal_vs_icnn_onestep.png", dpi=250)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(1000 * nominal["solve_time"], label="MPC N=20")
    plt.plot(1000 * icnn["solve_time"], label="one-step ICNN")
    plt.yscale("log")
    plt.xlabel("closed-loop step")
    plt.ylabel("solve time [ms]")
    plt.title("Solving time")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_nominal_vs_icnn_time.png", dpi=250)
    plt.close()


def plot_single_initial_state_performance(nominal, icnn, x0):
    t = np.arange(N_SIM + 1) * DT
    tu = np.arange(N_SIM) * DT

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, nominal["x"][:, 0], label="MPC N=20")
    plt.plot(t, icnn["x"][:, 0], "--", label="one-step ICNN")
    plt.ylabel("x1")
    plt.title(f"Single initial state, x0 = [{x0[0, 0]:.2f}, {x0[1, 0]:.2f}]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, nominal["x"][:, 1], label="MPC N=20")
    plt.plot(t, icnn["x"][:, 1], "--", label="one-step ICNN")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.step(tu, nominal["u"], where="post", label="MPC N=20")
    plt.step(tu, icnn["u"], where="post", linestyle="--", label="one-step ICNN")
    plt.axhline(U_BOUND, color="k", linestyle=":")
    plt.axhline(-U_BOUND, color="k", linestyle=":")
    plt.xlabel("time [s]")
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_single_initial_state_performance.png", dpi=250)
    plt.close()


def plot_single_initial_state_cost(nominal, icnn):
    tu = np.arange(N_SIM) * DT
    nominal_stage_cost = np.array(
        [stage_cost(nominal["x"][k].reshape(2, 1), nominal["u"][k]) for k in range(N_SIM)]
    )
    icnn_stage_cost = np.array(
        [stage_cost(icnn["x"][k].reshape(2, 1), icnn["u"][k]) for k in range(N_SIM)]
    )

    plt.figure(figsize=(9, 6))
    plt.subplot(2, 1, 1)
    plt.plot(tu, nominal_stage_cost, label="MPC N=20")
    plt.plot(tu, icnn_stage_cost, "--", label="one-step ICNN")
    plt.yscale("log")
    plt.ylabel("stage cost")
    plt.title("Cost performance, single initial state")
    plt.grid(True, which="both")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(tu, np.cumsum(nominal_stage_cost), label="MPC N=20")
    plt.plot(tu, np.cumsum(icnn_stage_cost), "--", label="one-step ICNN")
    plt.xlabel("time [s]")
    plt.ylabel("cumulative cost")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_single_initial_state_cost.png", dpi=250)
    plt.close()


def rollout_many_initial_states(value_predictor):
    results = []
    for idx, x0 in enumerate(COMPARISON_X0, start=1):
        print(f"Comparison rollout {idx}/{len(COMPARISON_X0)}: x0={x0.reshape(2)}")
        nominal = rollout_nominal_mpc(x0)
        icnn = rollout_one_step_icnn(x0, value_predictor)
        results.append({"x0": x0.copy(), "nominal": nominal, "icnn": icnn})
    return results


def rollout_many_icnn_initial_states(value_predictor):
    results = []
    for idx, x0 in enumerate(MULTI_X0, start=1):
        print(f"ICNN initial-state rollout {idx}/{len(MULTI_X0)}: x0={x0.reshape(2)}")
        results.append({"x0": x0.copy(), "icnn": rollout_one_step_icnn(x0, value_predictor)})
    return results


def plot_multi_initial_state_trajectories(results):
    t = np.arange(N_SIM + 1) * DT
    tu = np.arange(N_SIM) * DT

    plt.figure(figsize=(10, 7))
    for item in results:
        x0 = item["x0"].reshape(2)
        label_prefix = f"[{x0[0]:.1f}, {x0[1]:.1f}]"
        plt.plot(
            t,
            np.linalg.norm(item["nominal"]["x"], axis=1),
            linewidth=1.7,
            label=f"MPC {label_prefix}",
        )
        plt.plot(
            t,
            np.linalg.norm(item["icnn"]["x"], axis=1),
            "--",
            linewidth=1.5,
            label=f"ICNN {label_prefix}",
        )

    plt.xlabel("time [s]")
    plt.ylabel("||x||")
    plt.title("State norm from multiple initial states")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_multi_initial_state_norms.png", dpi=250)
    plt.close()

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    for item in results:
        x0 = item["x0"].reshape(2)
        label_prefix = f"[{x0[0]:.1f}, {x0[1]:.1f}]"
        axs[0].plot(t, item["nominal"]["x"][:, 0], label=f"MPC {label_prefix}")
        axs[0].plot(t, item["icnn"]["x"][:, 0], "--", label=f"ICNN {label_prefix}")
        axs[1].plot(t, item["nominal"]["x"][:, 1], label=f"MPC {label_prefix}")
        axs[1].plot(t, item["icnn"]["x"][:, 1], "--", label=f"ICNN {label_prefix}")
        axs[2].step(tu, item["nominal"]["u"], where="post", label=f"MPC {label_prefix}")
        axs[2].step(tu, item["icnn"]["u"], where="post", linestyle="--", label=f"ICNN {label_prefix}")

    axs[0].set_ylabel("x1")
    axs[0].set_title("x1 trajectories from multiple initial states")
    axs[1].set_ylabel("x2")
    axs[1].set_title("x2 trajectories from multiple initial states")
    axs[2].set_ylabel("u")
    axs[2].set_title("Control inputs from multiple initial states")
    axs[2].set_xlabel("time [s]")
    axs[2].axhline(U_BOUND, color="k", linestyle=":")
    axs[2].axhline(-U_BOUND, color="k", linestyle=":")
    for ax in axs:
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_multi_initial_state_trajectories.png", dpi=250)
    plt.close()


def plot_multi_initial_state_costs(results):
    t = np.arange(N_SIM) * DT
    plt.figure(figsize=(10, 7))

    for item in results:
        x0 = item["x0"].reshape(2)
        label_prefix = f"[{x0[0]:.1f}, {x0[1]:.1f}]"
        nominal_cost = np.cumsum(
            [stage_cost(item["nominal"]["x"][k].reshape(2, 1), item["nominal"]["u"][k]) for k in range(N_SIM)]
        )
        icnn_cost = np.cumsum(
            [stage_cost(item["icnn"]["x"][k].reshape(2, 1), item["icnn"]["u"][k]) for k in range(N_SIM)]
        )
        plt.plot(t, nominal_cost, label=f"MPC {label_prefix}")
        plt.plot(t, icnn_cost, "--", label=f"ICNN {label_prefix}")

    plt.xlabel("time [s]")
    plt.ylabel("cumulative cost")
    plt.title("Cumulative cost from multiple initial states")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_multi_initial_state_costs.png", dpi=250)
    plt.close()


def plot_icnn_multi_initial_time_series(results):
    t = np.arange(N_SIM + 1) * DT
    tu = np.arange(N_SIM) * DT
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(results)))

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#fbfbfd")
    for color, item in zip(colors, results):
        icnn = item["icnn"]
        axs[0].plot(t, icnn["x"][:, 0], color=color, linewidth=1.7, alpha=0.95)
        axs[1].plot(t, icnn["x"][:, 1], color=color, linewidth=1.7, alpha=0.95)
        axs[2].step(tu, icnn["u"], where="post", color=color, linewidth=1.35, alpha=0.95)

    axs[0].set_title("Stability of one-step MPC with ICNN cost approximation", pad=10)
    axs[0].set_ylabel(r"$x_1$")
    axs[1].set_ylabel(r"$x_2$")
    axs[2].set_ylabel(r"$u$")
    axs[2].set_xlabel("time [s]")
    axs[2].axhline(U_BOUND, color="#242424", linestyle=":", linewidth=1.2)
    axs[2].axhline(-U_BOUND, color="#242424", linestyle=":", linewidth=1.2)
    scalar = plt.cm.ScalarMappable(
        cmap=plt.cm.turbo,
        norm=plt.Normalize(0, len(results) - 1),
    )
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axs, pad=0.01, fraction=0.025)
    colorbar.set_label("initial state index")

    for ax in axs:
        ax.set_facecolor("white")
        ax.grid(True, color="#d8dbe2", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.savefig(OUT_DIR / "icnn_onestep_multi_initial_x1_x2_u.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_icnn_multi_initial_state_space(results):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    fig.patch.set_facecolor("#fbfbfd")
    ax.set_facecolor("white")

    time_norm = plt.Normalize(0.0, N_SIM * DT)
    cmap = plt.cm.viridis
    initial_colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(results)))

    for init_color, item in zip(initial_colors, results):
        traj = item["icnn"]["x"]
        points = traj[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line = LineCollection(segments, cmap=cmap, norm=time_norm, linewidth=2.8, alpha=0.95)
        line.set_array(np.arange(len(segments)) * DT)
        ax.add_collection(line)

        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            s=88,
            color=init_color,
            edgecolor="white",
            linewidth=1.4,
            zorder=4,
        )
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            s=52,
            marker="X",
            color="#111827",
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )

    ax.scatter(0.0, 0.0, s=170, marker="*", color="#ef4444", edgecolor="white", linewidth=1.2, zorder=6)
    ax.autoscale()
    ax.margins(0.08)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Stability of one-step MPC with ICNN cost approximation", pad=12)
    ax.grid(True, color="#d8dbe2", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=time_norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label("time [s]")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c026d3",
               markeredgecolor="white", markersize=9, label="initial states"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#111827",
               markeredgecolor="white", markersize=8, label="final states"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#ef4444",
               markeredgecolor="white", markersize=13, label="equilibrium"),
    ]
    ax.legend(handles=legend, loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "icnn_onestep_multi_initial_state_space.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_multi_initial_state_metrics(results):
    labels = [f"[{item['x0'][0, 0]:.1f},{item['x0'][1, 0]:.1f}]" for item in results]
    x_axis = np.arange(len(results))
    width = 0.36

    nominal_final = [np.linalg.norm(item["nominal"]["x"][-1]) for item in results]
    icnn_final = [np.linalg.norm(item["icnn"]["x"][-1]) for item in results]
    nominal_cost = [closed_loop_cost(item["nominal"]) for item in results]
    icnn_cost = [closed_loop_cost(item["icnn"]) for item in results]
    nominal_time = [1000 * item["nominal"]["solve_time"].mean() for item in results]
    icnn_time = [1000 * item["icnn"]["solve_time"].mean() for item in results]

    plt.figure(figsize=(11, 9))
    plt.subplot(3, 1, 1)
    plt.bar(x_axis - width / 2, nominal_final, width, label="MPC N=20")
    plt.bar(x_axis + width / 2, icnn_final, width, label="one-step ICNN")
    plt.yscale("log")
    plt.ylabel("final ||x||")
    plt.title("Closed-loop metrics from multiple initial states")
    plt.grid(True, axis="y", which="both")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.bar(x_axis - width / 2, nominal_cost, width, label="MPC N=20")
    plt.bar(x_axis + width / 2, icnn_cost, width, label="one-step ICNN")
    plt.ylabel("closed-loop cost")
    plt.grid(True, axis="y")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.bar(x_axis - width / 2, nominal_time, width, label="MPC N=20")
    plt.bar(x_axis + width / 2, icnn_time, width, label="one-step ICNN")
    plt.xticks(x_axis, labels)
    plt.xlabel("initial state")
    plt.ylabel("mean solve time [ms]")
    plt.grid(True, axis="y")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpc_multi_initial_state_metrics.png", dpi=250)
    plt.close()

    rows = []
    for label, item in zip(labels, results):
        rows.append(
            {
                "x0": label,
                "mpc_final_norm": np.linalg.norm(item["nominal"]["x"][-1]),
                "icnn_final_norm": np.linalg.norm(item["icnn"]["x"][-1]),
                "mpc_closed_loop_cost": closed_loop_cost(item["nominal"]),
                "icnn_closed_loop_cost": closed_loop_cost(item["icnn"]),
                "mpc_mean_solve_time_ms": 1000 * item["nominal"]["solve_time"].mean(),
                "icnn_mean_solve_time_ms": 1000 * item["icnn"]["solve_time"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "mpc_multi_initial_state_metrics.csv", index=False)


def print_summary(nominal, icnn):
    rows = [
        {
            "controller": f"MPC_N{N}",
            "final_norm": np.linalg.norm(nominal["x"][-1]),
            "closed_loop_stage_cost": closed_loop_cost(nominal),
            "mean_solve_time_ms": 1000 * nominal["solve_time"].mean(),
            "max_solve_time_ms": 1000 * nominal["solve_time"].max(),
        },
        {
            "controller": "one_step_ICNN",
            "final_norm": np.linalg.norm(icnn["x"][-1]),
            "closed_loop_stage_cost": closed_loop_cost(icnn),
            "mean_solve_time_ms": 1000 * icnn["solve_time"].mean(),
            "max_solve_time_ms": 1000 * icnn["solve_time"].max(),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mpc_nominal_vs_icnn_summary.csv", index=False)
    print("\n=== Closed-loop summary ===")
    print(df.to_string(index=False))


def main():
    print("Nominal model")
    print("A =\n", A_NOM)
    print("B =\n", B_NOM)
    print(f"N={N}, dt={DT}, u_bound={U_BOUND}, x0={X0.reshape(2)}")

    print("\nRunning nominal MPC...")
    nominal = rollout_nominal_mpc(X0)

    print("\nBuilding tail-cost dataset...")
    xs, values = build_tail_cost_dataset(force_rebuild=FORCE_REBUILD_DATASET)
    print(f"Dataset size: {len(xs)}")

    print("\nLoading/training ICNN...")
    model, x_mean, x_std, y_mean, y_std = load_or_train_icnn(xs, values, force_train=FORCE_RETRAIN_ICNN)
    value_predictor = make_value_predictor(model, x_mean, x_std, y_mean, y_std)

    print("\nRunning pure one-step ICNN MPC...")
    icnn = rollout_one_step_icnn(X0, value_predictor)

    plot_comparison(nominal, icnn)
    plot_single_initial_state_performance(nominal, icnn, X0)
    plot_single_initial_state_cost(nominal, icnn)

    print("\nRunning multi-initial-state performance plots...")
    icnn_multi_results = rollout_many_icnn_initial_states(value_predictor)
    plot_icnn_multi_initial_time_series(icnn_multi_results)
    plot_icnn_multi_initial_state_space(icnn_multi_results)

    multi_results = rollout_many_initial_states(value_predictor)
    plot_multi_initial_state_trajectories(multi_results)
    plot_multi_initial_state_costs(multi_results)
    plot_multi_initial_state_metrics(multi_results)

    print_summary(nominal, icnn)


if __name__ == "__main__":
    main()
