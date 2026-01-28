import numpy as np
import scipy.linalg as la
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from clqr import lmi_clqr  

# ==========================================================
# SYSTEM SETUP: ANTENNA MOTOR CONTROL, LTV system
# ==========================================================
A1 = np.array([[1, 0.1],
               [0, 0.9]])
A2 = np.array([[1, 0.1],
               [0, 0.1]])
B1 = np.array([[0],
               [0.1*0.787]])
B2 = np.array([[0],
               [0.1*0.787]])

A_list = [A1, A2]
B_list = [B1, B2]

n = 2
m = 1
Qc = np.eye(n)
Qc[0,0] = 1
Qc[1,1] = 0
R = np.array([[2e-5]])

dt = 0.1
Nsim = 200
u_bound = 2.0
y_bound = 1000.5
epsilon = 1e-5
BASE_DIR = Path(__file__).resolve().parent
figure_dir = BASE_DIR / "figure"
figure_dir.mkdir(exist_ok=True)
COLOR_RMPC = "tab:orange"
COLOR_LQR = "tab:green"
COLOR_ICNN = "tab:blue"


def positive_log_times(values):
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    if positive.size == 0:
        return np.full_like(values, 1e-9, dtype=float)
    floor = positive.min() * 0.5
    return np.where(values > 0, values, floor)


class ICNN_CLQR(nn.Module):
    def __init__(self, n_in=3, n_out=4, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64, 32]

        self.hidden_sizes = hidden_sizes
        self.Wz = nn.ModuleList()
        self.Wx = nn.ModuleList()

        for i, h in enumerate(hidden_sizes):
            if i == 0:
                self.Wz.append(nn.Linear(h, h, bias=False))
            else:
                self.Wz.append(nn.Linear(hidden_sizes[i - 1], h, bias=False))
            self.Wx.append(nn.Linear(n_in, h, bias=True))

        self.Wz_out = nn.Linear(hidden_sizes[-1], n_out, bias=False)
        self.Wx_out = nn.Linear(n_in, n_out, bias=True)

    def forward(self, x):
        z = torch.zeros(x.shape[0], self.hidden_sizes[0], device=x.device)
        for Wz, Wx in zip(self.Wz, self.Wx):
            z = F.relu(Wz(z) + F.relu(Wx(x)))
        return self.Wz_out(z) + self.Wx_out(x)

# Initial state
x = x0 = np.array([0.7, 0.7]).reshape(-1,1)
x_hist = x.copy()
u_hist = []
solving_time_clqr = []
poles_clqr = []
time_vec = [0.0]

vertex_id = 1  # Python index (0-based)

# ==========================================================
# MAIN SIMULATION LOOP: CLQR
# ==========================================================
for k in range(1, Nsim+1):
    print(f"=== Step {k} ===")
    t0 = time.perf_counter()
    Kclqr, gamma, Q, Y = lmi_clqr(A_list, B_list, Qc, R, x, u_bound, y_bound, epsilon)
    solving_time_clqr.append(time.perf_counter()-t0)

    if Kclqr is None:
        print(f"Infeasible LMI at step {k}")
        break

    # control input
    u = Kclqr @ x

    # simulate system (fixed vertex)
    Ai = A_list[vertex_id]
    Bi = B_list[vertex_id]
    x = Ai @ x + Bi @ u

    # store poles and history
    poles_clqr.append(np.linalg.eigvals(Ai + Bi @ Kclqr))
    x_hist = np.hstack([x_hist, x])
    u_hist.append(float(u.item()))
    time_vec.append(k*dt)

# ==========================================================
# STANDARD UNCONSTRAINED LQR
# ==========================================================
x = x0.copy()
x_hist_lqr = x.copy()
u_hist_lqr = []
poles_lqr = []
time_vec_lqr = [0.0]

A_d = A_list[0]
B_d = B_list[0]
S = la.solve_discrete_are(A_d, B_d, Qc, R)
Klqr = la.inv(B_d.T @ S @ B_d + R) @ (B_d.T @ S @ A_d)

for k in range(1, Nsim+1):
    u = -Klqr @ x
    u = np.clip(u, -u_bound, u_bound)

    Ai = A_list[vertex_id]
    Bi = B_list[vertex_id]
    x = Ai @ x + Bi @ u

    poles_lqr.append(np.linalg.eigvals(Ai + Bi @ Klqr))
    x_hist_lqr = np.hstack([x_hist_lqr, x])
    u_hist_lqr.append(float(u.item()))
    time_vec_lqr.append(k*dt)

# ==========================================================
# ICNN APPROXIMATION OF RMPC
# ==========================================================
x = x0.copy()
x_hist_icnn = x.copy()
u_hist_icnn = []
solving_time_icnn = []
time_vec_icnn = [0.0]

model = ICNN_CLQR()
model.load_state_dict(torch.load(BASE_DIR / "clqr_icnn_best_model.pth", map_location="cpu"))
model.eval()

for k in range(1, Nsim+1):
    x_in = np.array([[x[0, 0], x[1, 0], u_bound]], dtype=np.float32)
    x_in_tensor = torch.tensor(x_in)

    t0 = time.perf_counter()
    with torch.no_grad():
        y_pred = model(x_in_tensor).numpy()
    solving_time_icnn.append(time.perf_counter() - t0)

    u = float(np.clip(y_pred[0, 0], -u_bound, u_bound))

    Ai = A_list[vertex_id]
    Bi = B_list[vertex_id]
    x = Ai @ x + Bi @ np.array([[u]])

    x_hist_icnn = np.hstack([x_hist_icnn, x])
    u_hist_icnn.append(u)
    time_vec_icnn.append(k*dt)

# ==========================================================
# PLOTTING: STATES AND INPUTS
# ==========================================================
time_arr = np.array(time_vec)
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

time_arr_lqr = np.array(time_vec_lqr)
x_hist_lqr = np.array(x_hist_lqr)
u_hist_lqr = np.array(u_hist_lqr)

time_arr_icnn = np.array(time_vec_icnn)
x_hist_icnn = np.array(x_hist_icnn)
u_hist_icnn = np.array(u_hist_icnn)

plt.figure(figsize=(9,8))
plt.subplot(3,1,1)
plt.plot(time_arr, x_hist[0,:], label='x1 RMPC', linewidth=1.5, color=COLOR_RMPC)
plt.plot(time_arr_lqr, x_hist_lqr[0,:], label='x1 LQR', linewidth=1.5, color=COLOR_LQR)
plt.plot(time_arr_icnn, x_hist_icnn[0,:], label='x1 ICNN', linewidth=1.5, color=COLOR_ICNN)
plt.legend(); plt.grid(True); plt.title('System states: x1')

plt.subplot(3,1,2)
plt.plot(time_arr, x_hist[1,:], label='x2 RMPC', linewidth=1.5, color=COLOR_RMPC)
plt.plot(time_arr_lqr, x_hist_lqr[1,:], label='x2 LQR', linewidth=1.5, color=COLOR_LQR)
plt.plot(time_arr_icnn, x_hist_icnn[1,:], label='x2 ICNN', linewidth=1.5, color=COLOR_ICNN)
plt.legend(); plt.grid(True); plt.title('System states: x2')

plt.subplot(3,1,3)
plt.step(time_arr[:-1], u_hist, where='post', label='u RMPC', linewidth=1.5, color=COLOR_RMPC)
plt.step(time_arr_lqr[:-1], u_hist_lqr, where='post', label='u LQR', linewidth=1.5, color=COLOR_LQR)
plt.step(time_arr_icnn[:-1], u_hist_icnn, where='post', label='u ICNN', linewidth=1.5, color=COLOR_ICNN)
plt.axhline(u_bound, linestyle='--', color='k')
plt.axhline(-u_bound, linestyle='--', color='k')
plt.legend(); plt.grid(True); plt.title('Control input')
plt.tight_layout()
plt.savefig(figure_dir / "lqr_vs_clqr.png", dpi=300, bbox_inches="tight")
plt.savefig(figure_dir / "lqr_vs_rmpc_icnn.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
time_icnn = positive_log_times(solving_time_icnn)
time_rmpc = positive_log_times(solving_time_clqr)
mean_time_icnn = np.mean(time_icnn)
mean_time_rmpc = np.mean(time_rmpc)
plt.plot(1000*time_icnn, linewidth=1.5, marker='o', markersize=3, label='ICNN')
plt.plot(1000*time_rmpc, linewidth=1.5, marker='x', markersize=3, label='RMPC')
plt.axhline(1000*mean_time_icnn, color='C0', linestyle='--', linewidth=1.2,
            label=f'ICNN mean: {1000*mean_time_icnn:.3f} ms')
plt.axhline(1000*mean_time_rmpc, color='C1', linestyle='--', linewidth=1.2,
            label=f'RMPC mean: {1000*mean_time_rmpc:.3f} ms')
plt.xlabel('Iteration k'); plt.ylabel('[ms]')
plt.yscale('log')
plt.grid(True, which='both')
plt.legend()
plt.title('Computation time per step')
plt.tight_layout()
plt.savefig(figure_dir / "icnn_computational_time.png", dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# PLOTTING POLES CLQR
# ==========================================================
poles_clqr = np.array(poles_clqr).T  # shape (2,Nsim)
plt.figure()
plt.plot(np.real(poles_clqr[0,:]), np.imag(poles_clqr[0,:]), 'o-', label='Pole 1')
plt.plot(np.real(poles_clqr[1,:]), np.imag(poles_clqr[1,:]), 'o-', label='Pole 2')
theta = np.linspace(0, 2*np.pi, 400)
plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.5)  # unit circle
plt.xlabel('Real Axis'); plt.ylabel('Imaginary Axis')
plt.title('Pole Trajectories RMPC')
plt.grid(True); plt.axis('equal'); plt.legend()
plt.close()

# ==========================================================
# PLOTTING POLES LQR
# ==========================================================
poles_lqr = np.array(poles_lqr).T
plt.figure()
plt.plot(np.real(poles_lqr[0,:]), np.imag(poles_lqr[0,:]), 'o-', label='Pole 1')
plt.plot(np.real(poles_lqr[1,:]), np.imag(poles_lqr[1,:]), 'o-', label='Pole 2')
plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.5)
plt.xlabel('Real Axis'); plt.ylabel('Imaginary Axis')
plt.title('Pole Trajectories LQR')
plt.grid(True); plt.axis('equal'); plt.legend()
plt.close()
