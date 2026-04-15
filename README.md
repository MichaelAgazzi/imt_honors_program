# Robust Constrained MPC Using LMIs

This repository contains LMI-based robust Model Predictive Control material developed for an IMT Honors Program project on stability analysis, robust MPC, and learning-based control approximations.

The project was developed as part of the IMT honors program over the December-April period and later cleaned into this standalone repository.

The core reference is the included paper:

- `Robust_Constrained_MPC_using_LMI.pdf`: Kothare, Balakrishnan, and Morari, "Robust Constrained Model Predictive Control using Linear Matrix Inequalities", Automatica, 1996.

The examples study constrained linear systems with polytopic uncertainty, compute stabilizing feedback gains through Linear Matrix Inequalities, and compare robust constrained MPC/CLQR behavior against nominal LQR and neural approximations. The Python experiments connect this LMI-MPC material to ICNN value-function and Lyapunov-style approximations for reducing MPC online computation.

## Repository Structure

- `fig/`: selected MATLAB-generated figures for the constrained LQR/RMPC examples, including closed-loop responses and pole trajectories.
- `matlab/`: MATLAB/YALMIP implementations of LMI-based constrained control examples.
- `python/`: Python experiments for robust CLQR simulations, ICNN/MLP approximation, one-step MPC with learned tail cost, and Lyapunov-style ICNN controllers.
- `python/icnn_lyapunov/`: more structured PyTorch code for learning value/Lyapunov approximations and testing closed-loop behavior.
- `python/figure/`: plots used to document Python experiments, including CLQR/ICNN comparisons, model-selection diagnostics, computational-time summaries, and one-step MPC results.
- `Robust_Constrained_MPC_using_LMI.pdf`: included reference paper for the LMI robust MPC formulation.

## MATLAB Usage

Run the MATLAB examples from the repository root:

```matlab
run('matlab/main.m')
run('matlab/antenna_example.m')
run('matlab/antenna_traj_example.m')
```

The scripts add their own folder to the MATLAB path so helper functions such as `lmi_clqr.m` and `lmi_mpc.m` can be found when launched from the root.

MATLAB dependencies:

- MATLAB with Control System Toolbox functions such as `dlqr`.
- YALMIP.
- A semidefinite programming solver supported by YALMIP, such as SeDuMi, SDPT3, or MOSEK.

## Python Usage

Create and activate a Python environment, then install the main scientific stack:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy scipy pandas matplotlib torch cvxpy
```

Examples can be launched from the repository root:

```powershell
python .\python\sim_clqr_lqr.py
python .\python\mpc_icnn_onestep.py
python .\python\icnn_lyapunov\simulate_closed_loop.py --checkpoint icnn_lyapunov\gamma_icnn.pth --nsim 30 --u-max 2.0
```

The scripts use paths relative to their own source folder, so datasets, checkpoints, and generated figures remain under `python/`.

## Reproducing Examples

1. Start with the MATLAB LMI examples in `matlab/` to reproduce robust constrained feedback gains and closed-loop comparisons.
2. Use `python/sim_clqr_lqr.py` to compare LQR, robust CLQR/RMPC, and the stored ICNN approximation.
3. Use `python/mpc_icnn_onestep.py` to run the one-step MPC tail-cost approximation experiment. Existing datasets and checkpoints are reused unless the rebuild flags in the script are changed.
4. Use `python/icnn_lyapunov/README.md` for the more detailed ICNN Lyapunov/value-actor workflow.

## Stored Artifacts

The repository intentionally keeps small CSV datasets, trained `.pth` checkpoints, and generated figures that are needed to reproduce or inspect the existing examples without regenerating every optimization result from scratch. No tracked file is larger than 50 MB at the time of cleanup.

Future large experiment outputs should be kept out of Git and stored externally or regenerated from scripts.

See `NOTES.md` for dependency limitations and verification details. See
`DEVELOPMENT_TIMELINE.md` for the documented December-April development
chronology.

## Project Context

This project supports the IMT Honors Program work on robust constrained MPC, LMI stability conditions, and learning-based approximations. The LMI-MPC examples provide the robust-control foundation; the Python/ICNN material explores how convex neural approximations can reduce online computation while preserving value-function or Lyapunov-style structure where possible.
