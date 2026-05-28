import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
import pandas as pd



from clqr import lmi_clqr
from compare_conditions import check_upper_bound_per_vertex 

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
y_bound = 1000.0
epsilon = 1e-5
vertex_id = 1 


# create randomic initial states 
n_x0 = 5
x0_angle = 0.1
x0_angular_velocity = 0.1
delta_x_angle = np.pi/2
delta_x_angular_velocity = 1.0
x0_matrix = np.zeros((2,n_x0))
for i in range(n_x0):
    np.random.seed(i)
    x0_matrix[:,i] =  np.array([x0_angle + np.random.uniform(-delta_x_angle, delta_x_angle),
                   x0_angular_velocity + np.random.uniform(-delta_x_angular_velocity, delta_x_angular_velocity)]).reshape(-1)
    
# create randomic u_bound for each simulation
np.random.seed(0)
u_bound_values = u_bound + np.random.uniform(-1.0, 1.0, size=n_x0)
# ==========================================================
# plot x0 space and u space
f = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(x0_matrix[0,:], x0_matrix[1,:], label='Initial States')
plt.xlabel('Angle')
plt.ylabel('Angular Velocity')
plt.title('Initial States Distribution')
plt.grid()
plt.subplot(1,2,2)
plt.hist(u_bound_values, bins=10, color='orange', edgecolor='black')
plt.xlabel('u_bound values')
plt.ylabel('Frequency')
plt.title('u_bound Distribution')
plt.grid()
plt.tight_layout()
plt.show()

# create x_hist_all to store all simulations
x_hist_all = []
u_hist_all = []
u_bound_hist_all = []
gamma_hist_all = []
k_hist_all = []
comp_time_hist_all = []
ratio_y_x_all = []
x_plot = []
y_plot = []
z_plot = []
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
    gamma_hist = []
    ratio_y_x_hist = []
    solving_time_clqr = []
    poles_clqr = []
    time_vec = [0.0]
    
    # ==========================================================
    # SIMULATION LOOP: CLQR
    # ==========================================================
    for k in range(1, Nsim+1):
        # print(f"=== Step {k} ===")
        t0 = time.time()
        Kclqr, gamma, Q, Y = lmi_clqr(A_list, B_list, Qc, R, x, u_bound, y_bound, epsilon)
        solving_time_clqr.append(time.time()-t0)
        # compute ratio between norm Y and norm x
        ratio_y_x = np.linalg.norm(Y) / np.linalg.norm(x)
        ratio_y_x_hist.append(ratio_y_x)

        if Kclqr is None:
            print(f"Infeasible LMI at step {k}")
            break
        
        delta_vals, Psi_list, Xi_list, all_ok = check_upper_bound_per_vertex(A_list, B_list, Q, Y, gamma, Qc, R)
        delta_worst = np.min(delta_vals)   
        x_plot.append(x[0,0])
        y_plot.append(x[1,0])
        z_plot.append(delta_worst) 

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
        gamma_hist.append(gamma)
        u_bound_hist.append(u_bound)
        time_vec.append(k*dt)
        k_hist.append(Kclqr)
        
        
    # store state, gain and control for each initial condition
    x_hist_all.append(x_hist)
    u_hist_all.append(u_hist)
    k_hist_all.append(k_hist)
    ratio_y_x_all.append(ratio_y_x_hist)
    gamma_hist_all.append(gamma_hist)
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

    plt.figure(figsize=(12, 8))

    plt.subplot(5, 1, 1)
    plt.plot(time_vec, x_hist[0, :], label='Angle')
    plt.title(f'State Trajectories for Initial Condition {idx+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.legend()
    plt.grid()
    
    plt.subplot(5, 1, 2)
    plt.plot(time_vec, x_hist[1, :], label='Angular Velocity')
    plt.title(f'State Trajectories for Initial Condition {idx+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.legend()
    plt.grid()

    plt.subplot(5, 1, 3)
    plt.step(time_vec[:-1], u_hist, where='post')
    plt.title('Control Input Trajectory')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input u')
    plt.grid()
  
    plt.subplot(5, 1, 4)
    plt.plot(np.arange(len(comp_time_hist_all[idx])), comp_time_hist_all[idx], marker='o')
    plt.title('Computation Time per Step')
    plt.xlabel('Step')
    plt.ylabel('Time [s]')
    plt.grid()  

    plt.subplot(5, 1, 5)
    plt.plot(time_vec[:-1], gamma_hist_all[idx], marker='o')
    plt.title('Gamma Trajectory')
    plt.xlabel('Time [s]')
    plt.ylabel('Gamma')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(ratio_y_x_all[idx], marker='o')
    plt.title('Ratio between norm(Y) and norm(x) over time')
    plt.xlabel('Step')
    plt.ylabel('Ratio norm(Y)/norm(x)')
    plt.grid()
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
    data_dict[f'gamma_{idx+1}'] = gamma_hist_all[idx]
    
    k_array = np.array([K.flatten() for K in k_hist_all[idx]])  
    for k_idx in range(k_array.shape[1]):  
        data_dict[f'k{k_idx+1}_{idx+1}'] = k_array[:, k_idx]

    
# print(">>> Saving CSV...")
# df = pd.DataFrame(data_dict)
# df.to_csv(BASE_DIR / "dataset" / "simulation_data_gamma.csv", index=False)
# print(">>> CSV saved!")
# ==========================================================
# plot 3D trajectory of worst delta
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_plot, y_plot, z_plot, marker='o')
ax.set_xlabel('Angle')
ax.set_ylabel('Angular Velocity')
ax.set_zlabel('Worst delta value')
ax.set_title('upper bound > true equation')
# set limits for better visualization
ax.set_xlim([min(x_plot), max(x_plot)])
ax.set_ylim([min(y_plot), max(y_plot)])
ax.set_zlim([min(z_plot), max(z_plot)])
plt.show()
