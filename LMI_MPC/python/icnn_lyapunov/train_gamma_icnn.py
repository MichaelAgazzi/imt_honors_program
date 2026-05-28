from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from pathlib import Path
import sys

import torch
from torch import nn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from icnn_lyapunov.data import make_gamma_dataloaders
    from icnn_lyapunov.models import GammaICNN
else:
    from .data import make_gamma_dataloaders
    from .models import GammaICNN


def gamma_loss(
    gamma_pred: torch.Tensor,
    gamma_true: torch.Tensor,
    *,
    under_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = nn.functional.mse_loss(gamma_pred, gamma_true)
    over_penalty = torch.relu(gamma_pred - gamma_true).pow(2).mean()
    return mse + under_weight * over_penalty, mse, over_penalty


def evaluate(
    model: GammaICNN,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    under_weight: float,
) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    mse_sum = 0.0
    over_sum = 0.0
    max_over = 0.0
    with torch.no_grad():
        for x, gamma in loader:
            x = x.to(device)
            gamma = gamma.to(device)
            pred = model(x)
            loss, mse, over = gamma_loss(pred, gamma, under_weight=under_weight)
            batch_n = x.shape[0]
            total += batch_n
            loss_sum += float(loss.item()) * batch_n
            mse_sum += float(mse.item()) * batch_n
            over_sum += float(over.item()) * batch_n
            max_over = max(max_over, float(torch.relu(pred - gamma).max().item()))
    return {
        "loss": loss_sum / total,
        "mse": mse_sum / total,
        "over_penalty": over_sum / total,
        "max_over": max_over,
    }


def train(args: argparse.Namespace) -> GammaICNN:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    bundle = make_gamma_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        include_u_bound=args.include_u_bound,
        feature_mode=args.feature_mode,
    )
    model = GammaICNN(
        input_dim=bundle.x.shape[1],
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        enforce_positive_output=not args.allow_negative_output,
    ).to(device)
    model.set_input_normalization(bundle.input_mean.to(device), bundle.input_std.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, gamma in bundle.train_loader:
            x = x.to(device)
            gamma = gamma.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, _, _ = gamma_loss(pred, gamma, under_weight=args.under_weight)
            loss.backward()
            optimizer.step()
            model.project_nonnegative_weights()
            train_loss += float(loss.item()) * x.shape[0]
            n_train += x.shape[0]

        val_metrics = evaluate(model, bundle.val_loader, device=device, under_weight=args.under_weight)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch {epoch:04d} | train {train_loss / n_train:.6e} | "
                f"val {val_metrics['loss']:.6e} | val_mse {val_metrics['mse']:.6e} | "
                f"max_over {val_metrics['max_over']:.6e}"
            )
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.project_nonnegative_weights()

    if args.calibrate:
        x_cal = torch.stack([bundle.val_dataset[i][0] for i in range(len(bundle.val_dataset))]).to(device)
        y_cal = torch.stack([bundle.val_dataset[i][1] for i in range(len(bundle.val_dataset))]).to(device)
        shift = model.calibrate_lower_bound(x_cal, y_cal, safety_margin=args.calibration_margin)
        print(f"Applied lower-bound calibration shift: {shift:.6e}")

    test_metrics = evaluate(model, bundle.test_loader, device=device, under_weight=args.under_weight)
    print(
        f"test_loss {test_metrics['loss']:.6e} | test_mse {test_metrics['mse']:.6e} | "
        f"test_max_over {test_metrics['max_over']:.6e}"
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": bundle.x.shape[1],
            "hidden_sizes": tuple(args.hidden_sizes),
            "activation": args.activation,
            "enforce_positive_output": not args.allow_negative_output,
            "dataset": str(args.dataset),
            "feature_mode": bundle.feature_mode,
        },
        output_path,
    )
    print(f"Saved GammaICNN checkpoint to {output_path}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ICNN lower approximation of gamma*(x).")
    parser.add_argument("--dataset", default="simulation_data_gamma.csv")
    parser.add_argument("--output", default="icnn_lyapunov/gamma_icnn.pth")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--under-weight", type=float, default=10.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64, 32])
    parser.add_argument("--activation", choices=["relu", "softplus"], default="softplus")
    parser.add_argument(
        "--feature-mode",
        choices=["state", "state_u", "state_ubound", "theta"],
        default="state",
        help=(
            "Input features for GammaICNN: state=[x], state_u=[x,u], "
            "state_ubound=[x,u_bound], theta=[x,q11,q12,q22,y1,y2]."
        ),
    )
    parser.add_argument("--include-u-bound", action="store_true")
    parser.add_argument("--allow-negative-output", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-margin", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
