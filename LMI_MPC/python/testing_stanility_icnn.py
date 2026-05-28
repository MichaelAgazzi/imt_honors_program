# simulate_icnn_trajectories.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from motor import motor_step


class ICNN_CLQR(nn.Module):
    """
    ICNN per CLQR: garantisce convessità rispetto agli stati
    Input: x + u_bound -> n_in
    Output: u + K -> n_out
    """
    def __init__(self, n_in=3, n_out=4, hidden_sizes=[64, 32]):
        super(ICNN_CLQR, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes)

        # Layer hidden
        self.Wz = nn.ModuleList()
        self.Wx = nn.ModuleList()

        in_z = hidden_sizes[0]  
        for i, h in enumerate(hidden_sizes):
            if i == 0:
                self.Wz.append(nn.Linear(h, h, bias=False))
            else:
                self.Wz.append(nn.Linear(hidden_sizes[i-1], h, bias=False))
            self.Wx.append(nn.Linear(n_in, h, bias=True))
            # non negative initialozation
            nn.init.uniform_(self.Wx[-1].weight, a=0.0, b=0.1)

        # Output layer
        self.Wz_out = nn.Linear(hidden_sizes[-1], n_out, bias=False)
        self.Wx_out = nn.Linear(n_in, n_out, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        z = torch.zeros(batch_size, self.hidden_sizes[0], device=x.device)

        for i, (Wz, Wx) in enumerate(zip(self.Wz, self.Wx)):
            Wx_pos = F.relu(Wx(x))
            z = F.relu(Wz(z) + Wx_pos)

        y = self.Wz_out(z) + self.Wx_out(x)
        return y
    
# ===============================
# LOAD TRAINED MODEL
# ===============================
model_path = "clqr_icnn_best_model.pth"
model = ICNN_CLQR(n_in=3, n_out=4, hidden_sizes=[256, 128, 64]) 
model.load_state_dict(torch.load(model_path))
model.eval()

# ===============================
# SIMULATION GRID
# ===============================
x1_vals = np.linspace(-np.pi/2 + 0.001, np.pi/2 - 0.001, 50)
u_bounds = np.linspace(2.0, 2.0, 3)
x2_fixed = 0.5
Nsim = 250  # numero di step

# Creiamo un colormap per differenziare le condizioni iniziali
from matplotlib.cm import viridis
colors = viridis(np.linspace(0,1,len(x1_vals)*len(u_bounds)))

# Prepara figure
fig, axs = plt.subplots(3,1, figsize=(12,10), sharex=True)

for idx1, x1_0 in enumerate(x1_vals):
    for idx2, u_bound in enumerate(u_bounds):
        color_idx = idx1*len(u_bounds) + idx2
        color = colors[color_idx]
        
        x = np.array([[x1_0], [x2_fixed]])
        x1_traj = [x[0,0]]
        x2_traj = [x[1,0]]
        u_traj = []

        for k in range(Nsim):
            x_in = np.array([[x[0,0], x[1,0], u_bound]], dtype=np.float32)
            x_in_tensor = torch.tensor(x_in)
            with torch.no_grad():
                y_pred = model(x_in_tensor).numpy()
            u = y_pred[0,0]
            x = motor_step(x, u)
            
            x1_traj.append(x[0,0])
            x2_traj.append(x[1,0])
            u_traj.append(u)
        
        time_vec = np.arange(Nsim+1)  # step 0..Nsim
        axs[0].plot(time_vec, x1_traj, color=color, alpha=0.8)
        axs[1].plot(time_vec, x2_traj, color=color, alpha=0.8)
        axs[2].step(time_vec[:-1], u_traj, color=color, alpha=0.8, where='post')

# Labels and titles
axs[0].set_ylabel('x1')
axs[0].set_title('Trajectories of x1')
axs[0].grid(True)

axs[1].set_ylabel('x2')
axs[1].set_title('Trajectories of x2')
axs[1].grid(True)

axs[2].set_ylabel('u')
axs[2].set_xlabel('Step')
axs[2].set_title('Control input u')
axs[2].grid(True)

# Colorbar to show x1_0/u_bound mapping
sm = plt.cm.ScalarMappable(cmap='viridis', 
                           norm=plt.Normalize(vmin=0, vmax=len(x1_vals)*len(u_bounds)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('Condition index (x1_0 & u_bound combination)')

plt.tight_layout()
plt.show()

# ==========================================================
# CLQR TRAJECTORIES (same plot style as ICNN)
# ==========================================================
from clqr import lmi_clqr

# ----- CLQR parameters -----
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

Qc = np.eye(2)
Qc[0,0] = 1
Qc[1,1] = 0
R = np.array([[2e-5]])

y_bound = 1000.0
epsilon = 1e-5
vertex_id = 1

# ----- Colormap -----
from matplotlib.cm import viridis
colors = viridis(np.linspace(0, 1, len(x1_vals)*len(u_bounds)))

# ----- Figure -----
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# ==========================================================
# SIMULATION LOOP (CLQR)
# ==========================================================
for idx1, x1_0 in enumerate(x1_vals):
    for idx2, u_bound in enumerate(u_bounds):

        color_idx = idx1*len(u_bounds) + idx2
        color = colors[color_idx]

        x = np.array([[x1_0], [x2_fixed]])

        x1_traj = [x[0,0]]
        x2_traj = [x[1,0]]
        u_traj  = []

        for k in range(Nsim):
            Kclqr, gamma, Q, Y = lmi_clqr(
                A_list, B_list, Qc, R, x, u_bound, y_bound, epsilon
            )

            if Kclqr is None:
                break

            u = float(Kclqr @ x)
            x = motor_step(x, u, vertex_id=vertex_id)

            x1_traj.append(x[0,0])
            x2_traj.append(x[1,0])
            u_traj.append(u)

        time_vec = np.arange(len(x1_traj))

        axs[0].plot(time_vec, x1_traj, color=color, alpha=0.8)
        axs[1].plot(time_vec, x2_traj, color=color, alpha=0.8)
        axs[2].step(time_vec[:-1], u_traj, where='post', color=color, alpha=0.8)

# ==========================================================
# PLOT SETTINGS
# ==========================================================
axs[0].set_ylabel('x1')
axs[0].set_title('CLQR – Trajectories of x1')
axs[0].grid(True)

axs[1].set_ylabel('x2')
axs[1].set_title('CLQR – Trajectories of x2')
axs[1].grid(True)

axs[2].set_ylabel('u')
axs[2].set_xlabel('Step')
axs[2].set_title('CLQR – Control input u')
axs[2].grid(True)

# Colorbar
sm = plt.cm.ScalarMappable(
    cmap='viridis',
    norm=plt.Normalize(vmin=0, vmax=len(x1_vals)*len(u_bounds))
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.02)
cbar.set_label('Condition index (x1₀ & u_bound)')

plt.tight_layout()
plt.show()
