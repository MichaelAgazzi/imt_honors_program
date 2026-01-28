# neural network fitting CLQR controller
import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from motor import motor_step
from clqr import lmi_clqr  

BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# IMPORT DATASET
# ==========================================================
n = 2
m = 1

df = pd.read_csv(BASE_DIR / "simulation_data_gamma.csv")
print(df.head())

init_columns = [c for c in df.columns if "_init" in c]

init_indices = set([int(col.split("init")[1]) for col in init_columns])

n_x0 = max(init_indices)
print("Numero di condizioni iniziali:", n_x0)

Xs_list = []
Ub_list = []
U_list = []
T_list = []
K_list = []
Gamma_list = []


n = 2
m = 1

for idx in range(1, n_x0+1):

    X_i = np.vstack([df[f"x{i}_init{idx}"].values for i in range(1, n+1)])
    U_i = df[f"u{idx}"].values
    Ub_i = df[f"u_bound_{idx}"].values
    T_i = df[f"comp_time_{idx}"].values
    k1 = df[f'k1_{idx}'].values  # shape = (n_steps,)
    k2 = df[f'k2_{idx}'].values  # shape = (n_steps,)
    K_i = np.vstack([k1, k2]).T  # shape = (n_steps, 2)


    Xs_list.append(X_i)
    U_list.append(U_i)
    Ub_list.append(Ub_i)
    T_list.append(T_i)
    K_list.append(K_i)
    gamma_i = df[f'gamma_{idx}'].values  # shape = (n_steps,)
    Gamma_list.append(gamma_i)


    print(f"Init {idx}:   X: {X_i.shape}   U: {U_i.shape}   K: {K_i.shape}")

    
    
# ==========================================================
Xs_all = np.vstack([Xs_list[i].T for i in range(n_x0)])
U_all = np.hstack([U_list[i] for i in range(n_x0)])[:, None]
Ub_all = np.hstack([Ub_list[i] for i in range(n_x0)])[:, None]
K_all = np.vstack(K_list)  
Gamma_all = np.hstack(Gamma_list)[:, None]


print(Xs_all.shape, U_all.shape, K_all.shape, Gamma_all.shape)

# ==========================================================
Xs_tensor = torch.tensor(Xs_all, dtype=torch.float32)
Ub_tensor = torch.tensor(Ub_all, dtype=torch.float32)
U_tensor = torch.tensor(U_all, dtype=torch.float32)
K_tensor = torch.tensor(K_all, dtype=torch.float32)
Gamma_tensor = torch.tensor(Gamma_all, dtype=torch.float32)

Y_tensor = torch.cat([U_tensor, K_tensor, Gamma_tensor], dim=1)
X_tensor = torch.cat([Xs_tensor, Ub_tensor], dim=1)
print(X_tensor.shape, Y_tensor.shape)

# ==========================================================
# DATASET SPLIT TRAINING / VALIDATION / TEST
# ==========================================================
N = len(X_tensor)
train_size = int(0.7 * N)
val_size   = int(0.15 * N)
test_size  = N - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    torch.utils.data.TensorDataset(X_tensor, Y_tensor),
    [train_size, val_size, test_size]
)

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Total samples: {N}, Batches: {len(loader)}")
# ==========================================================

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
            # non negative initialization
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


# ==========================================================
# MODEL SELECTION SETUP
# ==========================================================

hidden_size_grid = [
    [64, 32],
    [128, 64, 32],
    [256, 128, 64],
    [256, 128, 64, 32]
]

n_epochs = 400
lr = 1e-3

best_val_loss = np.inf
best_model_state = None
best_arch = None

# store results for plots
arch_list = []
val_loss_list = []
train_loss_list = []

# ==========================================================
def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, n_epochs):

    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(n_epochs):

        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            # ICNN constraint: Wz >= 0
            with torch.no_grad():
                for layer in model.Wz:
                    layer.weight.clamp_(min=0.0)

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= train_size
        train_loss_hist.append(train_loss)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= val_size
        val_loss_hist.append(val_loss)

    return train_loss_hist, val_loss_hist, val_loss
# ==========================================================


# ==========================================================
# MODEL SELECTION LOOP
# ==========================================================

for hidden_sizes in hidden_size_grid:

    print(f"\nTraining ICNN with hidden sizes: {hidden_sizes}")

    model = ICNN_CLQR(
        n_in=3,
        n_out=4,
        hidden_sizes=hidden_sizes
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_hist, val_hist, final_val_loss = train_and_validate(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        n_epochs
    )

    print(f"Final validation loss: {final_val_loss:.6e}")

    arch_list.append(hidden_sizes)
    val_loss_list.append(final_val_loss)
    train_loss_list.append(train_hist[-1])

    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_model_state = model.state_dict()
        best_arch = hidden_sizes


print("\n===================================")
print("BEST MODEL FOUND")
print(f"Hidden sizes: {best_arch}")
print(f"Validation loss: {best_val_loss:.6e}")
print("===================================")

# ricrea il best model
best_model = ICNN_CLQR(
    n_in=3,
    n_out=4,
    hidden_sizes=best_arch
)
best_model.load_state_dict(best_model_state)

torch.save(best_model.state_dict(), BASE_DIR / "clqr_icnn_best_model.pth")
model = best_model

plt.figure(figsize=(7,4))
plt.plot(val_loss_list, marker='o')
plt.yscale('log')
plt.xlabel('Model index')
plt.ylabel('Validation MSE')
plt.title('Validation MSE for Different ICNN Architectures')
plt.grid(True)
plt.show()

n_layers_list = [len(arch) for arch in arch_list]

plt.figure(figsize=(7,4))
plt.scatter(n_layers_list, val_loss_list, s=80)
plt.yscale('log')
plt.xlabel('Number of hidden layers')
plt.ylabel('Validation MSE')
plt.title('Validation MSE vs Network Depth')
plt.grid(True)
plt.show()

n_neurons_list = [sum(arch) for arch in arch_list]

plt.figure(figsize=(7,4))
plt.scatter(n_neurons_list, val_loss_list, s=80)
plt.yscale('log')
plt.xlabel('Total number of hidden neurons')
plt.ylabel('Validation MSE')
plt.title('Validation MSE vs Model Capacity')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(val_loss_list, marker='o')
plt.yscale('log')
plt.xlabel('Model index')
plt.ylabel('Validation MSE')
plt.title('Validation MSE Across ICNN Architectures')
plt.grid(True)

for i, arch in enumerate(arch_list):
    plt.text(i, val_loss_list[i], str(arch), fontsize=9, rotation=45)

plt.tight_layout()
plt.show()

# ==========================================================

optimizer = optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

# ==========================================================
# TRAINING AND VALIDATION
# ==========================================================

n_epochs = 700
train_loss_hist = []
val_loss_hist = []

for epoch in range(n_epochs):

    # ===== TRAIN =====
    model.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for layer in model.Wz:
                layer.weight.clamp_(min=0.0)
        train_loss += loss.item() * batch_x.size(0)

    train_loss /= train_size
    train_loss_hist.append(train_loss)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            val_loss += loss.item() * batch_x.size(0)

    val_loss /= val_size
    val_loss_hist.append(val_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e}")
        
# save model 
torch.save(model.state_dict(), BASE_DIR / "clqr_icnn_model.pth")
# ==========================================================
plt.figure()
plt.plot(train_loss_hist, label='Train')
plt.plot(val_loss_hist, label='Validation')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training vs Validation Loss')
plt.grid()
plt.legend()
plt.show()

# ==========================================================
# TEST THE NETWORK
# ==========================================================
model.eval()
test_loss = 0.0
test_loss_hist = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        test_loss += loss.item() * batch_x.size(0)
        test_loss_hist.append(loss.item())

test_loss /= test_size
print(f"TEST LOSS: {test_loss:.6e}")

plt.figure()
plt.plot(test_loss_hist)
plt.xlabel('Batch') 
plt.ylabel('MSE')
plt.title(f'Test Loss per Batch - Avg Loss: {test_loss:.6e}')
plt.grid()
plt.show()


with torch.no_grad():
    Y_pred = model(X_tensor).numpy()

U_pred = Y_pred[:, 0]
K_pred = Y_pred[:, 1:3]
GAMMA_pred = Y_pred[:, 3]
mse_u = np.mean((U_pred - U_all.flatten())**2)
print(f"dim K_pred: {K_pred.shape}, dim K_all: {K_all.shape}")
mse_k1 = np.mean((K_pred[:, 0] - K_all[:, 0])**2)
mse_k2 = np.mean((K_pred[:, 1] - K_all[:, 1])**2)
mse_k = (mse_k1 + mse_k2)/2
mse_gamma = np.mean((GAMMA_pred - Gamma_all.flatten())**2)
print(f"MSE U: {mse_u:.6f}, MSE K: {mse_k:.6f}, MSE Gamma: {mse_gamma:.6f}")

plt.figure(figsize=(10,5))
plt.plot(U_all, label="U", marker='o')
plt.plot(U_pred, '--', label="U ICNN")
plt.legend()
plt.title(f"Comparison input U - MSE: {mse_u:.6f}")
plt.grid()
plt.show()


plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(K_all[:,0], label="K1")
plt.plot(K_pred[:,0], '--', label="K1 ICNN", marker='o')
plt.legend()
plt.title(f"Comparison gain K1 - MSE: {mse_k1:.6f}")
plt.grid()
plt.subplot(2,1,2)
plt.plot(K_all[:,1], label="K2")
plt.plot(K_pred[:,1], '--', label="K2 ICNN")
plt.legend()
plt.title(f"Comparison gain K2 - MSE: {mse_k2:.6f}")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(Gamma_all, label="Gamma")
plt.plot(GAMMA_pred, '--', label="Gamma ICNN")
plt.legend()
plt.title(f"Comparison Gamma - MSE: {mse_gamma:.6f}")
plt.grid()
plt.show()


# ==========================================================
# SIMULATE SYSTEM USING TRAINED NN CONTROLLER VS CLOQR
# ==========================================================
x0 = np.array([[-0.4], [-1.0]]) 
u_bound = 1.2
Nsim_test = 100
vertex_id = 1  

x_hist = x0.copy()
x = x0.copy()
u_hist = []
K_hist = []
Gamma_hist = []
comp_time_vec = []
for k in range(Nsim_test):
    # input model [x1, x2, u_bound]
    x_in = np.array([[x[0,0], x[1,0], u_bound]], dtype=np.float32)  
    x_in_tensor = torch.tensor(x_in)
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(x_in_tensor).numpy()
    comp_time_vec.append(time.time() - t0)
    u = y_pred[0, 0]         # first value = control u
    K_pred_step = y_pred[0, 1:3]  # gains K
    Gamma_pred_step = y_pred[0, 3]  # gamma 
    x = motor_step(x, u, vertex_id=vertex_id)  # step dynamics
    
    x_hist = np.hstack([x_hist, x])
    u_hist.append(u)
    K_hist.append(K_pred_step)
    Gamma_hist.append(Gamma_pred_step)
u_hist = np.array(u_hist)
K_hist = np.array(K_hist)
Gamma_hist = np.array(Gamma_hist)

# clqr
x_hist_clqr = x0.copy()
x = x0.copy()
u_hist_clqr = []
K_hist_clqr = []
Gamma_hist_clqr = []
comp_time_vec_clqr = []
Qc = np.eye(n)
Qc[0,0] = 1
Qc[1,1] = 0
R = np.array([[2e-5]])
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
y_bound = 1000.0
epsilon = 1e-5
for k in range(Nsim_test):
    t0 = time.time()
    Kclqr, gamma, Q, Y = lmi_clqr(A_list, B_list, Qc, R, x, u_bound, y_bound, epsilon)
    comp_time_vec_clqr.append(time.time() - t0)
    u = Kclqr @ x
    x = motor_step(x, u, vertex_id=vertex_id)  # step dynamics
    x_hist_clqr = np.hstack([x_hist_clqr, x])
    u_hist_clqr.append(u)
    K_hist_clqr.append(Kclqr.flatten())
    Gamma_hist_clqr.append(gamma)
u_hist_clqr = np.array(u_hist_clqr).reshape(-1)
K_hist_clqr = np.array(K_hist_clqr)
Gamma_hist_clqr = np.array(Gamma_hist_clqr)

# ==========================================================
# PLOT RESULTS
# ==========================================================
time_vec = np.arange(Nsim_test+1) * 0.1 

plt.figure(figsize=(12,6))
plt.subplot(4,1,1)
plt.plot(time_vec, x_hist[0,:], label='Angle ICNN')
plt.plot(time_vec, x_hist_clqr[0,:], '--', label='Angle CLQR')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.legend()
plt.grid()

plt.subplot(4,1,2)
plt.plot(time_vec, x_hist[1,:], label='Angular Velocity ICNN')
plt.plot(time_vec, x_hist_clqr[1,:], '--', label='Angular Velocity CLQR')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.legend()
plt.grid()


plt.subplot(4,1,3)
plt.step(time_vec[:-1], u_hist, where='post', label='Control input U')
plt.step(time_vec[:-1], u_hist_clqr, where='post', linestyle='--', label='Control input U CLQR')
plt.xlabel('Time [s]')
plt.ylabel('U')
plt.grid()
plt.legend()

plt.subplot(4,1,4)
plt.plot(K_hist[:,0], label='K1 NN', marker='o')
plt.plot(K_hist[:,1], label='K2 NN', marker='o')
plt.plot(K_hist_clqr[:,0], '--', label='K1 CLQR')
plt.plot(K_hist_clqr[:,1], '--', label='K2 CLQR')
plt.xlabel('Step')
plt.ylabel('K gains')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(Gamma_hist, label='Gamma NN', marker='x')
plt.plot(Gamma_hist_clqr, '--', label='Gamma CLQR', marker='o')
plt.xlabel('Step')
plt.ylabel('Gamma')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
# compute mean computation time per step
mean_time_ICNN = np.mean(comp_time_vec)
mean_time_clqr = np.mean(comp_time_vec_clqr)
plt.axhline(mean_time_ICNN, color='blue', linestyle='--', label=f'ICNN Mean Time: {mean_time_ICNN:.4f}s')
plt.axhline(mean_time_clqr, color='orange', linestyle='--', label=f'CLQR Mean Time: {mean_time_clqr:.4f}s')
plt.plot(np.arange(len(comp_time_vec)), comp_time_vec, marker='o')
plt.plot(np.arange(len(comp_time_vec_clqr)), comp_time_vec_clqr, marker='x')
plt.legend()
plt.title('Computation Time per Step')
plt.xlabel('Step')
plt.ylabel('Time [s]')
plt.grid()
plt.show()
