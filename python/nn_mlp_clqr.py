# neural network fitting CLQR controller
import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from motor import motor_step

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

# ==========================================================
# IMPORT DATASET
# ==========================================================
n = 2
m = 1

df = pd.read_csv(DATA_DIR / "simulation_data.csv")
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

    print(f"Init {idx}:   X: {X_i.shape}   U: {U_i.shape}   K: {K_i.shape}")

    
    
# ==========================================================
Xs_all = np.vstack([Xs_list[i].T for i in range(n_x0)])
U_all = np.hstack([U_list[i] for i in range(n_x0)])[:, None]
Ub_all = np.hstack([Ub_list[i] for i in range(n_x0)])[:, None]
K_all = np.vstack(K_list)  

print(Xs_all.shape, U_all.shape, K_all.shape)

# ==========================================================
Xs_tensor = torch.tensor(Xs_all, dtype=torch.float32)
Ub_tensor = torch.tensor(Ub_all, dtype=torch.float32)
U_tensor = torch.tensor(U_all, dtype=torch.float32)
K_tensor = torch.tensor(K_all, dtype=torch.float32)

Y_tensor = torch.cat([U_tensor, K_tensor], dim=1)
X_tensor = torch.cat([Xs_tensor, Ub_tensor], dim=1)
print(X_tensor.shape, Y_tensor.shape)
# ==========================================================
class CLQRNet(nn.Module):
    def __init__(self, n_in=3, n_out=3, h1=64, h2=32):
        super(CLQRNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_in, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_out)
        )

    def forward(self, x):
        return self.net(x)


model = CLQRNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

batch_size = 128
dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

n_epochs = 1000
loss_history = []

for epoch in range(n_epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
    loss_history.append(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")
        
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.grid()
plt.show()

# ==========================================================
# TEST THE NETWORK
# ==========================================================
with torch.no_grad():
    Y_pred = model(X_tensor).numpy()

U_pred = Y_pred[:, 0]
K_pred = Y_pred[:, 1:]
mse_u = np.mean((U_pred - U_all.flatten())**2)
print(f"dim K_pred: {K_pred.shape}, dim K_all: {K_all.shape}")
mse_k1 = np.mean((K_pred[:, 0] - K_all[:, 0])**2)
mse_k2 = np.mean((K_pred[:, 1] - K_all[:, 1])**2)
mse_k = (mse_k1 + mse_k2)/2

plt.figure(figsize=(10,5))
plt.plot(U_all, label="U reale")
plt.plot(U_pred, '--', label="U predetto")
plt.legend()
plt.title(f"Confronto input U - MSE: {mse_u:.6f}")
plt.grid()
plt.show()


plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(K_all[:,0], label="K1 reale")
plt.plot(K_pred[:,0], '--', label="K1 predetto")
plt.legend()
plt.title(f"Confronto guadagno K1 - MSE: {mse_k1:.6f}")
plt.grid()
plt.subplot(2,1,2)
plt.plot(K_all[:,1], label="K2 reale")
plt.plot(K_pred[:,1], '--', label="K2 predetto")
plt.legend()
plt.title(f"Confronto guadagno K2 - MSE: {mse_k2:.6f}")
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# SIMULATE SYSTEM USING TRAINED NN CONTROLLER
# ==========================================================
x0 = np.array([[0.4], [0.4]]) 
Nsim_test = 100
vertex_id = 1  

x_hist = x0.copy()
x = x0.copy()
u_hist = []
K_hist = []
comp_time_vec = []
import time
for k in range(Nsim_test):
    # prepara input per il modello: [x1, x2, u_bound]
    x_in = np.array([[x[0,0], x[1,0], 2.0]], dtype=np.float32)  # u_bound = 2.0
    x_in_tensor = torch.tensor(x_in)
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(x_in_tensor).numpy()
    comp_time_vec.append(time.time() - t0)
    u = y_pred[0, 0]         # primo valore = controllo u
    K_pred_step = y_pred[0, 1:]  # resto = guadagni K
    
    x = motor_step(x, u, vertex_id=vertex_id)  # passo dinamica
    
    x_hist = np.hstack([x_hist, x])
    u_hist.append(u)
    K_hist.append(K_pred_step)

u_hist = np.array(u_hist)
K_hist = np.array(K_hist)

# ==========================================================
# PLOT RESULTS
# ==========================================================
time_vec = np.arange(Nsim_test+1) * 0.1 

plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(time_vec, x_hist[0,:], label='Angle')
plt.plot(time_vec, x_hist[1,:], label='Angular Velocity')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.legend()
plt.grid()

plt.subplot(3,1,2)
plt.step(time_vec[:-1], u_hist, where='post', label='Control input U')
plt.xlabel('Time [s]')
plt.ylabel('U')
plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.plot(K_hist[:,0], label='K1 predetto')
plt.plot(K_hist[:,1], label='K2 predetto')
plt.xlabel('Step')
plt.ylabel('K gains')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(np.arange(len(comp_time_vec)), comp_time_vec, marker='o')
plt.title('Computation Time per Step')
plt.xlabel('Step')
plt.ylabel('Time [s]')
plt.grid()
plt.show()
