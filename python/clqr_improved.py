import numpy as np
import cvxpy as cp
import scipy.linalg as la

def lmi_clqr(
    A_list, B_list, C_list,
    Qc, R,
    xk,
    u_bound,
    y_bound_list,
    epsilon=1e-6,
    rho=1e3
):
    """
    Robust CLQR via LMIs with:
      - polytopic uncertainty
      - hard input constraints
      - soft componentwise output constraints

    Parameters
    ----------
    A_list, B_list : list of system vertices
    C_list         : list of output row matrices C_l (shape 1 x n)
    Qc, R          : LQR weights
    xk             : current state
    u_bound        : scalar input bound |u| <= u_bound
    y_bound_list   : list of bounds y_l
    epsilon        : numerical regularization
    rho            : soft constraint penalty

    Returns
    -------
    K, gamma, Q_val, Y_val, S_val
    """

    n = A_list[0].shape[0]
    m = B_list[0].shape[1]
    n_y = len(C_list)

    # ===============================
    # Decision variables
    # ===============================
    Q = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))
    gamma = cp.Variable(nonneg=True)
    S = cp.Variable(n_y, nonneg=True)   # soft slacks for output constraints

    constraints = []
    constraints += [Q >> epsilon * np.eye(n)]
    constraints += [gamma >= epsilon]

    # ===============================
    # Initial condition constraint
    # ===============================
    LMI_state = cp.bmat([
        [np.array([[1.0]]), xk.T],
        [xk, Q]
    ])
    constraints += [LMI_state >> 0]

    # ===============================
    # LQR performance (gamma)
    # ===============================
    Qc_eps = Qc + 1e-8 * np.eye(n)
    R_eps = R + 1e-12 * np.eye(m)

    sqrt_Qc = la.sqrtm(Qc_eps)
    sqrt_R = la.sqrtm(R_eps)

    if np.iscomplexobj(sqrt_Qc):
        sqrt_Qc = np.real_if_close(sqrt_Qc)
    if np.iscomplexobj(sqrt_R):
        sqrt_R = np.real_if_close(sqrt_R)

    # ===============================
    # Stability + performance LMIs
    # ===============================
    for Ai, Bi in zip(A_list, B_list):

        E = sqrt_Qc @ Q
        F = sqrt_R @ Y

        zeros_n_n = np.zeros((n, n))
        zeros_n_m = np.zeros((n, m))
        zeros_m_n = np.zeros((m, n))

        LMI = cp.bmat([
            [Q,               Q @ Ai.T + Y.T @ Bi.T,  E.T,              F.T],
            [Ai @ Q + Bi @ Y, Q,                      zeros_n_n,       zeros_n_m],
            [E,               zeros_n_n,              gamma * np.eye(n), zeros_n_m],
            [F,               zeros_m_n,              zeros_m_n,       gamma * np.eye(m)]
        ])

        constraints += [LMI >> 0]

    # ===============================
    # Hard input constraints
    # ===============================
    for Ai, Bi in zip(A_list, B_list):

        input_block = cp.bmat([
            [u_bound**2 * np.eye(m), Y],
            [Y.T, Q]
        ])
        constraints += [input_block >> 0]

    # ===============================
    # SOFT componentwise output constraints
    # ===============================
    for Ai, Bi in zip(A_list, B_list):

        AQ_by = Ai @ Q + Bi @ Y

        for l, (Cl, yb) in enumerate(zip(C_list, y_bound_list)):

            Cl = Cl.reshape(1, -1)   # ensure shape (1, n)

            out_block = cp.bmat([
                [Q,                   (Cl @ AQ_by).T],
                [Cl @ AQ_by,          (yb**2 + S[l]) * np.ones((1, 1))]
            ])

            constraints += [out_block >> 0]

    # ===============================
    # Objective
    # ===============================
    objective = cp.Minimize(gamma + rho * cp.sum(S))
    prob = cp.Problem(objective, constraints)

    # ===============================
    # Solve
    # ===============================
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception as e:
        print("Solver MOSEK failed:", e)
        return None, None, None, None, None

    if prob.status in ["optimal", "optimal_inaccurate"]:

        Q_val = Q.value
        Y_val = Y.value
        gamma_val = gamma.value
        S_val = S.value

        try:
            K = Y_val @ la.inv(Q_val)
        except la.LinAlgError:
            K = None

        return K, float(gamma_val), Q_val, Y_val, S_val

    else:
        print("Problem infeasible:", prob.status)
        return None, None, None, None, None
