from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

try:
    from clqr import lmi_clqr
    from motor import A_list, B_list
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from clqr import lmi_clqr
    from motor import A_list, B_list


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sample_initial_conditions(
    *,
    n_x0: int,
    seed: int,
    angle_center: float = 0.1,
    velocity_center: float = 0.1,
    angle_radius: float = np.pi / 2,
    velocity_radius: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x1 = angle_center + rng.uniform(-angle_radius, angle_radius, size=n_x0)
    x2 = velocity_center + rng.uniform(-velocity_radius, velocity_radius, size=n_x0)
    return np.vstack([x1, x2])


def generate_theta_dataset(
    *,
    output: str | Path,
    n_x0: int = 20,
    nsim: int = 5,
    seed: int = 3,
    u_bound: float = 2.0,
    u_bound_variation: float = 0.4,
    y_bound: float = 1000.0,
    epsilon: float = 1e-5,
    vertex_id: int = 1,
) -> pd.DataFrame:
    """Generate a compact dataset with x, theta=(Q,Y), u, K and gamma.

    The output is a long CSV, one row per feasible time step:

        x1, x2, q11, q12, q22, y1, y2, gamma

    are the features/target needed to train GammaICNN as gamma(x, theta).
    """
    n = 2
    q_c = np.eye(n)
    q_c[0, 0] = 1.0
    q_c[1, 1] = 0.0
    r = np.array([[2e-5]])

    x0_matrix = sample_initial_conditions(n_x0=n_x0, seed=seed)
    rng = np.random.default_rng(seed + 100)
    u_bounds = u_bound + rng.uniform(-u_bound_variation, u_bound_variation, size=n_x0)

    rows: list[dict[str, float | int]] = []
    sample_id = 0
    for traj_id in range(n_x0):
        x = x0_matrix[:, traj_id].reshape(2, 1)
        traj_u_bound = float(max(1e-3, u_bounds[traj_id]))
        print(
            f"trajectory {traj_id + 1:03d}/{n_x0:03d} | "
            f"x0=[{x[0, 0]: .3f}, {x[1, 0]: .3f}] | u_bound={traj_u_bound:.3f}"
        )

        for k in range(nsim):
            t0 = time.time()
            k_clqr, gamma, q_val, y_val = lmi_clqr(
                A_list,
                B_list,
                q_c,
                r,
                x,
                traj_u_bound,
                y_bound,
                epsilon,
            )
            solve_time = time.time() - t0
            if k_clqr is None or gamma is None or q_val is None or y_val is None:
                print(f"  infeasible at step {k}; stopping this trajectory")
                break

            q_val = np.asarray(q_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float).reshape(1, 2)
            k_clqr = np.asarray(k_clqr, dtype=float).reshape(1, 2)
            u = float((k_clqr @ x).item())
            u_sat = float(np.clip(u, -traj_u_bound, traj_u_bound))
            x_next = A_list[vertex_id] @ x + B_list[vertex_id] * u_sat

            rows.append(
                {
                    "sample_id": sample_id,
                    "traj_id": traj_id,
                    "k": k,
                    "x1": float(x[0, 0]),
                    "x2": float(x[1, 0]),
                    "u_bound": traj_u_bound,
                    "gamma": float(gamma),
                    "u": u_sat,
                    "k1": float(k_clqr[0, 0]),
                    "k2": float(k_clqr[0, 1]),
                    "q11": float(q_val[0, 0]),
                    "q12": float(q_val[0, 1]),
                    "q22": float(q_val[1, 1]),
                    "y1": float(y_val[0, 0]),
                    "y2": float(y_val[0, 1]),
                    "x1_next": float(x_next[0, 0]),
                    "x2_next": float(x_next[1, 0]),
                    "solve_time": solve_time,
                }
            )
            sample_id += 1
            x = x_next

    if not rows:
        raise RuntimeError("No feasible rows generated; check MOSEK/CVXPY setup or loosen bounds.")

    df = pd.DataFrame(rows)
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = _project_root() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact CLQR theta=(Q,Y) dataset.")
    parser.add_argument("--output", default="icnn_lyapunov/simulation_data_theta_small.csv")
    parser.add_argument("--n-x0", type=int, default=20)
    parser.add_argument("--nsim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--u-bound", type=float, default=2.0)
    parser.add_argument("--u-bound-variation", type=float, default=0.4)
    parser.add_argument("--y-bound", type=float, default=1000.0)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--vertex-id", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_theta_dataset(
        output=args.output,
        n_x0=args.n_x0,
        nsim=args.nsim,
        seed=args.seed,
        u_bound=args.u_bound,
        u_bound_variation=args.u_bound_variation,
        y_bound=args.y_bound,
        epsilon=args.epsilon,
        vertex_id=args.vertex_id,
    )


if __name__ == "__main__":
    main()
