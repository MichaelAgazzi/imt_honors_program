# Project Notes

## Repository Scope

This repository is a standalone organization of the LMI MPC material from the
broader IMT Honors Program working directory. The work was developed during the
December-April project period and was subsequently arranged for sharing,
inspection, and reproducibility.

The retained material includes MATLAB LMI examples, Python simulations,
trained checkpoints, compact datasets, selected figures, and the reference paper
used for the robust constrained MPC formulation.

The historical development chronology is summarized in
`DEVELOPMENT_TIMELINE.md`.

## Dependencies

MATLAB examples require:

- MATLAB.
- Control System Toolbox functions such as `dlqr`.
- YALMIP.
- An SDP solver supported by YALMIP, such as SeDuMi, SDPT3, or MOSEK.

Python examples require:

- Python 3.10 or newer.
- `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, and `cvxpy`.
- A CVXPY solver suitable for the examples, typically CLARABEL or OSQP.

## Reproducibility

- Full MATLAB example execution was not performed during repository organization because it can
  depend on the local YALMIP and SDP solver configuration.
- Full Python experiments were not rerun because several scripts solve repeated
  optimization problems and train neural networks for many iterations.
- Python syntax checks were run with `python -m compileall -q python`.
- MATLAB path resolution was checked by asking MATLAB to locate
  `matlab/lmi_clqr.m` from the repository root.
- Existing datasets, trained checkpoints, and figures are kept because they make
  the examples inspectable without regenerating all optimization/training output.
- No tracked file is larger than 50 MB at the time of repository organization.
- No files outside `push_only_LMI_MPC` were modified.
