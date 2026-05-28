import numpy as np
import cvxpy as cp
import scipy.linalg as la

def clqr_vertex(A, B, Qc, R, x0, u_bound=10.0, y_bound=100.0, epsilon=1e-6):
    """
    Risolve CLQR per un singolo vertice usando LMI come in lmi_clqr
    Restituisce:
        K   : guadagno ottimale
        P   : matrice di Lyapunov (Q^-1)
        Q_i : Q
        Y_i : Y = K*Q
    """
    n = A.shape[0]
    m = B.shape[1]
    epsilon2 = 1e-3

    # Variabili CVXPY
    Q = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    gamma = cp.Variable(nonneg=True)

    constraints = []
    constraints += [Q >> epsilon*np.eye(n)]
    constraints += [gamma >= epsilon]

    # LMI_state: minimizza energia
    LMI_state = cp.bmat([[np.array([[1.0]]), x0.T],
                         [x0, Q]])
    constraints += [LMI_state >> 0]

    # Piccola regolarizzazione
    Qc_eps = Qc + 1e-8*np.eye(n)
    R_eps = R + 1e-12*np.eye(m)

    sqrt_Qc = la.sqrtm(Qc_eps)
    sqrt_R = la.sqrtm(R_eps)
    sqrt_Qc = np.real_if_close(sqrt_Qc)
    sqrt_R = np.real_if_close(sqrt_R)

    # LMI stabilità singolo vertice
    E = sqrt_Qc @ Q
    F = sqrt_R @ Y
    zeros_n_m = np.zeros((n, m))
    zeros_m_n = np.zeros((m, n))

    LMI = cp.bmat([
        [Q,         Q@A.T + Y.T@B.T,  E.T,         F.T],
        [A@Q + B@Y, Q,                  np.zeros((n,n)), zeros_n_m],
        [E,         np.zeros((n,n)),    gamma*np.eye(n), zeros_n_m],
        [F,         np.zeros((m,n)),    np.zeros((m,n)), gamma*np.eye(m)]
    ])
    constraints += [LMI >> epsilon2*np.eye(LMI.shape[0])]

    # Vincolo input
    input_block = cp.bmat([[u_bound**2*np.eye(m), Y],
                           [Y.T, Q]])
    constraints += [input_block >> 0]

    # Vincolo stato/output
    AQ_by = A @ Q + B @ Y
    so_block = cp.bmat([[Q, AQ_by.T],
                        [AQ_by, y_bound**2*np.eye(n)]])
    constraints += [so_block >> 0]

    # Problema LMI
    prob = cp.Problem(cp.Minimize(gamma), constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception as e:
        print("Solver failed:", e)
        return None, None, None, None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        Q_val = Q.value
        Y_val = Y.value
        gamma_val = gamma.value
        try:
            K = Y_val @ la.inv(Q_val)
        except la.LinAlgError:
            K = None
        return K, la.inv(Q_val), Q_val, Y_val
    else:
        print("Problem infeasible:", prob.status)
        return None, None, None, None