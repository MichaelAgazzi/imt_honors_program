import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pandas as pd


from clqr_improved import lmi_clqr  

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
DATA_DIR.mkdir(exist_ok=True)

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
C_list = [np.array([[1, 0]]), np.array([[0, 1]])]
y_bound_list = [0.1, 1.0]

n = 2
m = 1
Qc = np.eye(n)
Qc[0,0] = 1
Qc[1,1] = 0
R = np.array([[2e-5]])

dt = 0.1
Nsim = 100
u_bound = 15.0
epsilon = 1e-5
vertex_id = 1 


# create randomic initial states 
n_x0 = 1
x0_angle = 1.0
x0_angular_velocity = 0.0
delta_x_angle = 0.0
delta_x_angular_velocity = 0.0
x0_matrix = np.zeros((2,n_x0))
for i in range(n_x0):
    np.random.seed(i)
    x0_matrix[:,i] =  np.array([x0_angle + np.random.uniform(-delta_x_angle, delta_x_angle),
                   x0_angular_velocity + np.random.uniform(-delta_x_angular_velocity, delta_x_angular_velocity)]).reshape(-1)
    
# create randomic u_bound for each simulation
np.random.seed(0)
u_bound_values = u_bound + np.random.uniform(-0.5, 0.5, size=n_x0)
# ==========================================================
    
# create x_hist_all to store all simulations
x_hist_all = []
u_hist_all = []
u_bound_hist_all = []
S_list_all = []
k_hist_all = []
comp_time_hist_all = []
# ==========================================================
# SIMULATION FOR MULTIPLE INITIAL STATES
# ==========================================================
for i in range(n_x0):
    # Initial state
    print(f"--- Simulation for initial condition {i+1}/{n_x0} ---")
    x = x0 = np.array([x0_matrix[0,i], x0_matrix[1,i]]).reshape(-1,1)
    x_hist = x.copy()
    u_bound = u_bound_values[i]
    
    u_hist = []
    u_bound_hist = []
    k_hist = []
    S_list = []
    solving_time_clqr = []
    poles_clqr = []
    time_vec = [0.0]
    


    # ==========================================================
    # SIMULATION LOOP: CLQR
    # ==========================================================
    for k in range(1, Nsim+1):
        print(f"=== Step {k} ===")
        t0 = time.time()
        Kclqr, gamma, Q, Y, S = lmi_clqr(A_list, B_list, C_list, Qc, R, x, u_bound, y_bound_list, epsilon, rho=1e6)
        solving_time_clqr.append(time.time()-t0)

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
        u_hist.append(float(u))
        u_bound_hist.append(u_bound)
        S_list.append(S)
        time_vec.append(k*dt)
        k_hist.append(Kclqr)
        
    # store state, gain and control for each initial condition
    x_hist_all.append(x_hist)
    u_hist_all.append(u_hist)
    k_hist_all.append(k_hist)
    S_list_all.append(S_list)   
    u_bound_hist_all.append(u_bound_hist)
    comp_time_hist_all.append(solving_time_clqr)

#==========================================================
# PLOTTING FOR THE FIRST INITIAL CONDITION and the LAST ONE
#==========================================================

for idx in [0, n_x0-1]:
    x_hist = x_hist_all[idx]
    u_hist = u_hist_all[idx]
    k_hist = k_hist_all[idx]

    time_vec = np.array(time_vec)

    plt.figure()
    # find value such that y<y_bound
    indices = np.where(x_hist[0, :] < y_bound_list[0])[0]
    S1 = [S[0] for S in S_list_all[idx]]
    S2 = [S[1] for S in S_list_all[idx]]
    indeces_S0 = np.where(np.array(S1) < 1e-6)[0]
    plt.subplot(2,1,1)
    plt.plot(np.arange(len(comp_time_hist_all[idx])), S1, marker='o')
    plt.axvline(x=indices[0], color='r', linestyle='--', label='vertical line')
    plt.axvline(x=indeces_S0[0], color='g', linestyle='--', label='vertical line S1>0')
    plt.title('Soft Slack Variable S1 over Time')
    plt.xlabel('Step')
    plt.ylabel('S1 value')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(np.arange(len(comp_time_hist_all[idx])), S2, marker='o')
    plt.title('Soft Slack Variable S2 over Time')
    plt.xlabel('Step')
    plt.ylabel('S2 value')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(time_vec, x_hist[0, :], label='Angle')
    plt.axvline(x=indices[0]*dt, color='r', linestyle='--', label=' constraints respected')
    plt.axhline(y_bound_list[0], color='g', linestyle='--', label='y bound')
    plt.title(f'State Trajectories for Initial Condition {idx+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.legend()
    plt.grid()
    
    plt.subplot(4, 1, 2)
    plt.plot(time_vec, x_hist[1, :], label='Angular Velocity')
    plt.title(f'State Trajectories for Initial Condition {idx+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.step(time_vec[:-1], u_hist, where='post')
    plt.title('Control Input Trajectory')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input u')
    plt.grid()
  
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(len(comp_time_hist_all[idx])), comp_time_hist_all[idx], marker='o')
    plt.title('Computation Time per Step')
    plt.xlabel('Step')
    plt.ylabel('Time [s]')
    plt.grid()  
    plt.tight_layout()
    plt.show()
    
    
    

    
# ==========================================================
# save as csv data
data_dict = {}
for idx in range(n_x0):
    n_steps = len(u_hist_all[idx])

    for state_idx in range(n):
        data_dict[f'x{state_idx+1}_init{idx+1}'] = x_hist_all[idx][state_idx, :n_steps]

    data_dict[f'u{idx+1}'] = u_hist_all[idx]
    data_dict[f'comp_time_{idx+1}'] = comp_time_hist_all[idx]
    data_dict[f'u_bound_{idx+1}'] = u_bound_hist_all[idx]

    k_array = np.array([K.flatten() for K in k_hist_all[idx]])  
    for k_idx in range(k_array.shape[1]):  
        data_dict[f'k{k_idx+1}_{idx+1}'] = k_array[:, k_idx]

    
print(">>> Saving CSV...")
df = pd.DataFrame(data_dict)
df.to_csv(DATA_DIR / 'simulation_data_dif_u_bound.csv', index=False)
print(">>> CSV saved!")
# ==========================================================
