import numpy as np
import matplotlib.pyplot as plt
from clqr_vertex import clqr_vertex
from clqr_alpha import clqr_alpha

# =========================================
# SYSTEM
# =========================================
A1 = np.array([[1, 0.1],[0,0.9]])
A2 = np.array([[1, 0.1],[0,0.1]])
A3 = np.array([[1, 1.1],[0.2,0.5]])
B3 = np.array([[0],[0.1*0.787]])
B1 = np.array([[0],[0.1*0.787]])
B2 = np.array([[0],[0.1*0.787]])

A_list = [A1, A2, A3]
B_list = [B1, B2, B3]

Qc = np.eye(2)
Qc[0,0] = 1
Qc[1,1] = 0
R = np.array([[2e-5]])

x0 = np.array([[0.01],[0.01]])
u_bound = 10.0
y_bound = 500.0
# =========================================
# Step 1: CLQR per ogni vertice
# =========================================
K_list = []
P_list = []
Q_list = []
Y_list = []

for Ai, Bi in zip(A_list, B_list):
    K, P, Q_i, Y_i = clqr_vertex(Ai, Bi, Qc, R, x0, u_bound, y_bound)
    print("Closed-loop eigenvalues:", np.linalg.eigvals(Ai - Bi @ K))
    print("K:\n", K)
    K_list.append(K)
    P_list.append(P)
    Q_list.append(Q_i)
    Y_list.append(Y_i)
    
    
from clqr import lmi_clqr 

# =========================================
# Solve LMI CLQR
# =========================================
K, gamma, Q_val, Y_val = lmi_clqr(A_list, B_list, Qc, R, x0, u_bound, y_bound, epsilon= 1e-6)

if K is None:
    print("Problem infeasible: cannot compute K")
else:
    print("Computed LMI-CLQR K:\n", K)
    print("Gamma:", gamma)

    # =========================================
    # SIMULATION
    # =========================================
    Nsim = 50
    x = x0.copy()
    x_hist = [x]
    u_hist = []

    for k in range(Nsim):
        u = K @ x
        u = np.clip(u, -u_bound, u_bound)  # clip input

        # Randomly select a vertex
        vertex = np.random.choice(len(A_list))
        A = A_list[vertex]
        B = B_list[vertex]

        x = A @ x + B @ u
        x_hist.append(x)
        u_hist.append(u)

    x_hist = np.hstack(x_hist)
    u_hist = np.array(u_hist).flatten()

    # =========================================
    # PLOT
    # =========================================
    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.plot(x_hist[0,:], label='State x1 (angle)')
    plt.plot(x_hist[1,:], label='State x2 (angular velocity)')
    plt.ylabel('States')
    plt.grid(True)
    plt.legend()

    plt.subplot(2,1,2)
    plt.step(range(len(u_hist)), u_hist, label='Control input', where='post')
    plt.ylabel('Control input')
    plt.xlabel('Time step')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    

# =========================================
# Step 2: CLQR parametrico alpha
# =========================================
K_alpha, alpha_val, Q_alpha, Y_alpha = clqr_alpha(A_list, B_list, Q_list, Y_list, x0, u_bound, y_bound)

print("Alpha values:", alpha_val)
print("K_alpha:\n", K_alpha)

# =========================================
# Step 3: simulazione
# =========================================
x = x0.copy()
x_hist = [x]
u_hist = []

for k in range(50):
    u = K_alpha @ x
    # simulo cambiando verice randomicamente
    vertex = np.random.choice(len(A_list))
    A = A_list[vertex]
    B = B_list[vertex]
    x = A @ x + B @ u
    x_hist.append(x)
    u_hist.append(u)

x_hist = np.hstack(x_hist)
u_hist = np.array(u_hist).flatten()

plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(x_hist[0,:], label='Angle')
plt.plot(x_hist[1,:], label='Angular velocity')
plt.legend(); plt.grid()
plt.subplot(2,1,2)
plt.step(range(len(u_hist)), u_hist, label='Control input', where='post')
plt.grid(); plt.show()

# print alpha values and K_alpha
print("Alpha values:", alpha_val)
print("K_alpha:\n", K_alpha)


