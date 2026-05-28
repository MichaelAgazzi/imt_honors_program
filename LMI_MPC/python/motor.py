# motor.py
import numpy as np

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

# ==========================================================
def motor_step(x, u, vertex_id=1):
    """
    Simula un singolo passo della dinamica del motore.
    
    Args:
        x (np.array): stato corrente (n x 1)
        u (float or np.array): input di controllo
        vertex_id (int): indice del vertice del sistema (default 1)
        
    Returns:
        x_next (np.array): stato successivo
    """
    Ai = A_list[vertex_id]
    Bi = B_list[vertex_id]
    x_next = Ai @ x + Bi * u 
    return x_next
