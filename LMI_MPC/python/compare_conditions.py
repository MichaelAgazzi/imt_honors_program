import numpy as np
import scipy.linalg as la

def check_upper_bound_per_vertex(A_list, B_list, Q, Y, gamma, Qc, R):
    """
    Calcola Ψ e Ξ per ogni vertice separatamente e verifica se Ψ ⪰ Ξ.

    Restituisce:
      delta_vals : array dei minimi autovalori Psi_i - Xi_i per vertice
      Psi_list   : lista di Ψ vertice per vertice
      Xi_list    : lista di Xi_i per vertice
      all_ok     : True se Ψ ⪰ Ξ per tutti i vertici, False se almeno un vertice viola
    """
    delta_vals = []
    Psi_list = []
    Xi_list = []

    Qinv = la.inv(Q)
    P = gamma * Qinv
    F = Y @ Qinv

    for Ai, Bi in zip(A_list, B_list):
        # Closed-loop dynamics per vertice
        Acl = Ai + Bi @ F
        eigs = np.linalg.eigvals(Acl)
        lam_vert = np.max(np.abs(eigs))  # λ massimo per questo vertice

        # Ψ specifico per il vertice
        Psi = - Qc - F.T @ R @ F
        Xi = Acl.T @ P @ Acl - P - F.T @ R @ F - Qc

        Delta = Psi - Xi
        delta_vals.append(np.min(np.linalg.eigvals(Delta).real))
        Psi_list.append(Psi)
        Xi_list.append(Xi)

    delta_vals = np.array(delta_vals)
    all_ok = np.all(delta_vals > 0)  # True se Ψ ⪰ Ξ per tutti i vertici

    return delta_vals, Psi_list, Xi_list, all_ok
