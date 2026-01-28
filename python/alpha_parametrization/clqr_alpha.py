import numpy as np
import cvxpy as cp

def clqr_alpha(A_list, B_list, Q_list, Y_list, x0, u_bound, y_bound, epsilon=1e-6):
    """
    Risolve CLQR parametrizzando:
        Q = sum_i α_i Q_i
        Y = sum_i α_i Y_i
    con sum(α_i) = 1, α_i >= 0
    """
    n = A_list[0].shape[0]
    m = B_list[0].shape[1]
    N = len(A_list)

    alpha = cp.Variable(2*N)
    constraints = [alpha >= 0, cp.sum(alpha) == 1]

    # Interpolated Q and Y
    Q = sum(alpha[i] * Q_list[i] for i in range(N))
    Y = sum(alpha[i] * Y_list[i] for i in range(N))

    # Vincolo positivo definito
    constraints += [Q >> epsilon*np.eye(n)]

    # LMI stabilità per tutti i vertici
    for Ai, Bi in zip(A_list, B_list):
        LMI = cp.bmat([
            [Q, (Ai@Q + Bi@Y).T],
            [Ai@Q + Bi@Y, Q]
        ])
        constraints += [LMI >> epsilon*np.eye(2*n)]

    # Vincolo input
    for Bi in B_list:
        input_block = cp.bmat([
            [u_bound**2*np.eye(m), Y],
            [Y.T, Q]
        ])
        constraints += [input_block >> 0]

    # Minimizzare qualche gamma (qui gamma = trace(Q) come esempio)
    gamma = cp.Variable(nonneg=True)
    constraints += [gamma >= epsilon]
    prob = cp.Problem(cp.Minimize(gamma), constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    if prob.status in ["optimal", "optimal_inaccurate"]:
        alpha_val = alpha.value
        Q_val = sum(alpha_val[i] * Q_list[i] for i in range(N))
        Y_val = sum(alpha_val[i] * Y_list[i] for i in range(N))
        K_val = Y_val @ np.linalg.inv(Q_val)
        return K_val, alpha_val, Q_val, Y_val
    else:
        print("Problem infeasible:", prob.status)
        return None, None, None, None