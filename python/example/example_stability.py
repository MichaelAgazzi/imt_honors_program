import numpy as np
import matplotlib.pyplot as plt
from matplotlib import get_backend
from matplotlib.rcsetup import non_interactive_bk

# ============================================================
# 1. Dominio e vera Lyapunov V*
# ============================================================

Omega_min, Omega_max = -2.0, 2.0


def finalize_figure():
    """Show figures only on interactive backends; otherwise close them cleanly."""
    backend = get_backend().lower()
    if backend in {name.lower() for name in non_interactive_bk}:
        plt.close()
    else:
        plt.show()

def V_star(x):
    # Lyapunov convessa vera
    return x**2 + 0.15*x**4

def V_hat(x):
    return V_star(x) + 0.08*np.sin(5*x) - 0.03*x

# ============================================================
# 2. Griglia densa per visualizzare il ground truth
# ============================================================

x_dense = np.linspace(Omega_min, Omega_max, 2000)
v_true = V_star(x_dense)
v_pred = V_hat(x_dense)
err = v_true - v_pred

# ============================================================
# 3. Griglia di validazione più rada
# ============================================================

M = 25
z = np.linspace(Omega_min, Omega_max, M)

e_z = V_star(z) - V_hat(z)

beta_minus_emp = np.max(e_z)      # max(V* - Vhat)
beta_plus_emp  = np.max(-e_z)     # max(Vhat - V*)

# ============================================================
# 4. Stima data-driven del Lipschitz di V*
# ============================================================

# Differenze finite sui punti z
LV_emp = 0.0
for i in range(M):
    for j in range(i+1, M):
        ratio = abs(V_star(z[i]) - V_star(z[j])) / abs(z[i] - z[j])
        LV_emp = max(LV_emp, ratio)

kappa = 1.5
LV_cert = kappa * LV_emp

# Lipschitz della rete stimato numericamente sulla griglia densa
grad_hat = np.gradient(V_hat(x_dense), x_dense)
Lhat_cert = np.max(np.abs(grad_hat))

# Raggio massimo di copertura della griglia
h = z[1] - z[0]
rho = h / 2

margin = (LV_cert + Lhat_cert) * rho

beta_minus_cert = beta_minus_emp + margin
beta_plus_cert  = beta_plus_emp + margin

print("beta_minus_emp =", beta_minus_emp)
print("beta_plus_emp  =", beta_plus_emp)
print("LV_emp         =", LV_emp)
print("LV_cert        =", LV_cert)
print("Lhat_cert      =", Lhat_cert)
print("rho            =", rho)
print("margin         =", margin)
print("beta_minus_cert =", beta_minus_cert)
print("beta_plus_cert  =", beta_plus_cert)

# ============================================================
# 5. Plot V* e Vhat
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(x_dense, v_true, label=r"$V^*(x)$")
plt.plot(x_dense, v_pred, label=r"$\hat V(x)$")
plt.scatter(z, V_star(z), marker="o", label="punti validazione")
plt.xlabel("x")
plt.ylabel("V")
plt.title("Vera Lyapunov e approssimazione neurale")
plt.legend()
plt.grid(True)
finalize_figure()

# ============================================================
# 6. Plot errore e bound certificati
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(x_dense, err, label=r"$e(x)=V^*(x)-\hat V(x)$")
plt.scatter(z, e_z, label="errore sui punti di validazione")

plt.axhline(beta_minus_emp, linestyle="--", label=r"$\hat\beta_-^{emp}$")
plt.axhline(-beta_plus_emp, linestyle="--", label=r"$-\hat\beta_+^{emp}$")

plt.axhline(beta_minus_cert, linestyle="-.", label=r"$\beta_-^{cert}$")
plt.axhline(-beta_plus_cert, linestyle="-.", label=r"$-\beta_+^{cert}$")

plt.xlabel("x")
plt.ylabel("errore")
plt.title("Errore osservato e bound certificati")
plt.legend()
plt.grid(True)
finalize_figure()

# ============================================================
# 7. Visualizzazione del tubo certificato
# ============================================================

upper_cert = V_hat(x_dense) + beta_minus_cert
lower_cert = V_hat(x_dense) - beta_plus_cert

plt.figure(figsize=(10, 5))
plt.plot(x_dense, v_true, label=r"$V^*(x)$")
plt.plot(x_dense, v_pred, label=r"$\hat V(x)$")
plt.fill_between(
    x_dense,
    lower_cert,
    upper_cert,
    alpha=0.25,
    label="tubo certificato"
)

plt.xlabel("x")
plt.ylabel("V")
plt.title(r"Tubo certificato: $\hat{V}-\beta_+ \leq V^* \leq \hat{V}+\beta_-$")
plt.legend()
plt.grid(True)
finalize_figure()
