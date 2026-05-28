from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset


_INIT_RE = re.compile(r"x1_init(\d+)$")


@dataclass(frozen=True)
class GammaDatasetBundle:
    x: torch.Tensor
    gamma: torch.Tensor
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: TensorDataset
    val_dataset: TensorDataset
    test_dataset: TensorDataset
    input_mean: torch.Tensor
    input_std: torch.Tensor
    feature_mode: str


@dataclass(frozen=True)
class ValueActorDatasetBundle:
    x: torch.Tensor
    u_star: torch.Tensor
    v_star: torch.Tensor
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: TensorDataset
    val_dataset: TensorDataset
    test_dataset: TensorDataset
    input_mean: torch.Tensor
    input_std: torch.Tensor
    value_column: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _discover_indices(columns: list[str]) -> list[int]:
    indices = []
    for column in columns:
        match = _INIT_RE.match(column)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def load_gamma_samples(
    csv_path: str | Path | None = None,
    *,
    n_state: int = 2,
    include_u_bound: bool = False,
    feature_mode: str = "state",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load state/gamma pairs from the CLQR simulation CSV layout.

    The expected columns are x1_init{i}, ..., xn_init{i} and gamma_{i}. If the
    requested file has no gamma columns, the function raises a clear error
    instead of silently training on the wrong target.
    """
    if include_u_bound:
        feature_mode = "state_ubound"
    valid_modes = {"state", "state_u", "state_ubound", "theta"}
    if feature_mode not in valid_modes:
        raise ValueError(f"feature_mode must be one of {sorted(valid_modes)}, got {feature_mode!r}.")

    path = Path(csv_path) if csv_path is not None else _project_root() / "simulation_data_gamma.csv"
    if not path.is_absolute():
        path = _project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    long_theta_columns = {"x1", "x2", "q11", "q12", "q22", "y1", "y2", "gamma"}
    if feature_mode == "theta" and long_theta_columns.issubset(df.columns):
        feature_columns = ["x1", "x2", "q11", "q12", "q22", "y1", "y2"]
        block = df[feature_columns + ["gamma"]].dropna()
        if block.empty:
            raise ValueError(f"No complete theta samples found in {path}")
        return (
            torch.tensor(block[feature_columns].to_numpy(), dtype=torch.float32),
            torch.tensor(block[["gamma"]].to_numpy(), dtype=torch.float32),
        )

    long_state_columns = {"x1", "x2", "gamma"}
    if feature_mode in {"state", "state_u"} and long_state_columns.issubset(df.columns):
        feature_columns = ["x1", "x2"]
        if feature_mode == "state_u":
            if "u" not in df.columns:
                raise ValueError(f"{path.name} has long format but no u column for state_u mode.")
            feature_columns.append("u")
        block = df[feature_columns + ["gamma"]].dropna()
        if block.empty:
            raise ValueError(f"No complete long-format samples found in {path}")
        return (
            torch.tensor(block[feature_columns].to_numpy(), dtype=torch.float32),
            torch.tensor(block[["gamma"]].to_numpy(), dtype=torch.float32),
        )

    indices = _discover_indices(list(df.columns))
    if not indices:
        raise ValueError(f"No x*_init columns found in {path}")

    gamma_columns = [column for column in df.columns if column.startswith("gamma_")]
    if not gamma_columns:
        fallback = _project_root() / "simulation_data_gamma.csv"
        raise ValueError(
            f"{path.name} does not contain gamma_* target columns. "
            f"Use {fallback} or regenerate the requested CSV including gamma from the SDP solver."
        )

    x_rows = []
    gamma_rows = []
    for idx in indices:
        state_columns = [f"x{i}_init{idx}" for i in range(1, n_state + 1)]
        gamma_column = f"gamma_{idx}"
        control_column = f"u{idx}"
        ubound_column = f"u_bound_{idx}"
        required_columns = state_columns + [gamma_column]
        if feature_mode == "state_u":
            required_columns.append(control_column)
        elif feature_mode == "state_ubound":
            required_columns.append(ubound_column)
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            continue

        feature_columns = list(state_columns)
        if feature_mode == "state_u":
            feature_columns.append(control_column)
        elif feature_mode == "state_ubound":
            feature_columns.append(ubound_column)
        block = df[feature_columns + [gamma_column]].dropna()
        if block.empty:
            continue
        x_rows.append(torch.tensor(block[feature_columns].to_numpy(), dtype=torch.float32))
        gamma_rows.append(torch.tensor(block[[gamma_column]].to_numpy(), dtype=torch.float32))

    if not x_rows:
        raise ValueError(f"No complete state/gamma samples found in {path}")

    return torch.cat(x_rows, dim=0), torch.cat(gamma_rows, dim=0)


def make_gamma_dataloaders(
    csv_path: str | Path | None = None,
    *,
    batch_size: int = 128,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 7,
    include_u_bound: bool = False,
    feature_mode: str = "state",
) -> GammaDatasetBundle:
    if include_u_bound:
        feature_mode = "state_ubound"
    x, gamma = load_gamma_samples(
        csv_path,
        include_u_bound=include_u_bound,
        feature_mode=feature_mode,
    )

    input_mean = x.mean(dim=0)
    input_std = x.std(dim=0, unbiased=False).clamp_min(1e-8)
    dataset = TensorDataset(x, gamma)
    n_samples = len(dataset)
    train_size = int(train_fraction * n_samples)
    val_size = int(val_fraction * n_samples)
    test_size = n_samples - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Dataset split produced an empty split.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    return GammaDatasetBundle(
        x=x,
        gamma=gamma,
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_mean=input_mean,
        input_std=input_std,
        feature_mode=feature_mode,
    )


def _first_existing(columns: set[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_value_actor_samples(
    csv_path: str | Path | None = None,
    *,
    n_state: int = 2,
    value_columns: Sequence[str] = ("V_star", "v_star", "value", "cost_to_go", "gamma"),
    control_columns: Sequence[str] = ("u_star", "u", "control"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Load ``(x, u_star, V_star)`` samples for actor/value training.

    The compact motor CSV generated by ``generate_theta_dataset.py`` has
    ``x1``, ``x2``, ``u`` and ``gamma``. In that case ``gamma`` is interpreted
    as the available optimal value/Lyapunov target.
    """
    path = Path(csv_path) if csv_path is not None else _project_root() / "simulation_data_theta_small.csv"
    if not path.is_absolute():
        path = _project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    columns = set(df.columns)

    long_state_columns = [f"x{i}" for i in range(1, n_state + 1)]
    value_column = _first_existing(columns, value_columns)
    control_column = _first_existing(columns, control_columns)
    if set(long_state_columns).issubset(columns) and value_column is not None and control_column is not None:
        block = df[long_state_columns + [control_column, value_column]].dropna()
        if block.empty:
            raise ValueError(f"No complete value/actor samples found in {path}")
        return (
            torch.tensor(block[long_state_columns].to_numpy(), dtype=torch.float32),
            torch.tensor(block[[control_column]].to_numpy(), dtype=torch.float32),
            torch.tensor(block[[value_column]].to_numpy(), dtype=torch.float32),
            value_column,
        )

    indices = _discover_indices(list(df.columns))
    if not indices:
        raise ValueError(
            f"{path.name} must contain either long-format x1/x2/u/gamma columns "
            "or wide x*_init{i}, u{i}, gamma_{i} columns."
        )

    x_rows = []
    u_rows = []
    v_rows = []
    used_value_column = ""
    for idx in indices:
        state_columns = [f"x{i}_init{idx}" for i in range(1, n_state + 1)]
        control_column = f"u{idx}"
        candidate_value_columns = [f"{name}_{idx}" for name in value_columns] + [f"gamma_{idx}"]
        value_column_idx = _first_existing(columns, candidate_value_columns)
        required = state_columns + [control_column]
        if value_column_idx is None or any(column not in columns for column in required):
            continue
        block = df[state_columns + [control_column, value_column_idx]].dropna()
        if block.empty:
            continue
        x_rows.append(torch.tensor(block[state_columns].to_numpy(), dtype=torch.float32))
        u_rows.append(torch.tensor(block[[control_column]].to_numpy(), dtype=torch.float32))
        v_rows.append(torch.tensor(block[[value_column_idx]].to_numpy(), dtype=torch.float32))
        used_value_column = value_column_idx

    if not x_rows:
        raise ValueError(f"No complete value/actor samples found in {path}")

    return torch.cat(x_rows, dim=0), torch.cat(u_rows, dim=0), torch.cat(v_rows, dim=0), used_value_column


def make_value_actor_dataloaders(
    csv_path: str | Path | None = None,
    *,
    batch_size: int = 128,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 7,
) -> ValueActorDatasetBundle:
    x, u_star, v_star, value_column = load_value_actor_samples(csv_path)

    input_mean = x.mean(dim=0)
    input_std = x.std(dim=0, unbiased=False).clamp_min(1e-8)
    dataset = TensorDataset(x, u_star, v_star)
    n_samples = len(dataset)
    if n_samples < 3:
        return ValueActorDatasetBundle(
            x=x,
            u_star=u_star,
            v_star=v_star,
            train_loader=DataLoader(dataset, batch_size=batch_size, shuffle=True),
            val_loader=DataLoader(dataset, batch_size=batch_size, shuffle=False),
            test_loader=DataLoader(dataset, batch_size=batch_size, shuffle=False),
            train_dataset=dataset,
            val_dataset=dataset,
            test_dataset=dataset,
            input_mean=input_mean,
            input_std=input_std,
            value_column=value_column,
        )
    train_size = int(train_fraction * n_samples)
    val_size = int(val_fraction * n_samples)
    test_size = n_samples - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Dataset split produced an empty split.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    return ValueActorDatasetBundle(
        x=x,
        u_star=u_star,
        v_star=v_star,
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_mean=input_mean,
        input_std=input_std,
        value_column=value_column,
    )
