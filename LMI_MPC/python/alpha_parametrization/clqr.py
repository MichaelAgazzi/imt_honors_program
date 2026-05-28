import numpy as np
import cvxpy as cp
import scipy.linalg as la

def lmi_clqr(A_list, B_list, Qc, R, xk, u_bound, y_bound, epsilon):
    """
    Solve LMI-based CLQR using CVXPY with MOSEK solver.
    Returns K, gamma, Q_val, Y_val.
    If infeasible returns (None, None, None, None)
    """
    epsilon2 = 0.001;
    n = A_list[0].shape[0]
    m = B_list[0].shape[1]

    # CVXPY variables
    Q = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    gamma = cp.Variable(nonneg=True)

    constraints = []
    constraints += [Q >> epsilon*np.eye(n)]
    constraints += [gamma >= epsilon]

    # LMI_state: minimize energy
    LMI_state = cp.bmat([[np.array([[1.0]]), xk.T],
                         [xk, Q]])
    constraints += [LMI_state >> 0]

    # Add a tiny regularization to avoid sqrtm warning
    Qc_eps = Qc + 1e-8*np.eye(n)
    R_eps = R + 1e-12*np.eye(m)

    sqrt_Qc = la.sqrtm(Qc_eps)
    sqrt_R = la.sqrtm(R_eps)

    if np.iscomplexobj(sqrt_Qc):
        sqrt_Qc = np.real_if_close(sqrt_Qc)
    if np.iscomplexobj(sqrt_R):
        sqrt_R = np.real_if_close(sqrt_R)

    # LMI for stability (for each vertex)
    for Ai, Bi in zip(A_list, B_list):
        E = sqrt_Qc @ Q
        F = sqrt_R @ Y

        zeros_n_m = np.zeros((n, m))
        zeros_m_n = np.zeros((m, n))

        LMI = cp.bmat([
            [Q,        Q@Ai.T + Y.T@Bi.T,  E.T,         F.T],
            [Ai@Q+Bi@Y, Q,                  np.zeros((n,n)), zeros_n_m],
            [E,         np.zeros((n,n)),    gamma*np.eye(n), zeros_n_m],
            [F,         np.zeros((m,n)),    np.zeros((m,n)), gamma*np.eye(m)]
        ])
        constraints += [LMI >> epsilon2*np.eye(LMI.shape[0])]

    # Input and state/output constraints
    for Ai, Bi in zip(A_list, B_list):
        input_block = cp.bmat([[u_bound**2*np.eye(m), Y],
                               [Y.T, Q]])
        constraints += [input_block >> 0]

        AQ_by = Ai @ Q + Bi @ Y
        so_block = cp.bmat([[Q, AQ_by.T],
                            [AQ_by, y_bound**2*np.eye(n)]])
        constraints += [so_block >> 0]

    # Objective: minimize gamma
    prob = cp.Problem(cp.Minimize(gamma), constraints)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception as e:
        print("Solver MOSEK failed:", e)
        return None, None, None, None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        Q_val = Q.value
        Y_val = Y.value
        gamma_val = gamma.value
        try:
            K = Y_val @ la.inv(Q_val)
        except la.LinAlgError:
            K = None
        return K, float(gamma_val), Q_val, Y_val
    else:
        print("Problem infeasible:", prob.status)
        return None, None, None, None
