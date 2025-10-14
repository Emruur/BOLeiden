import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ------------------------------------------------------------
# 1) Reuse GP and discrete set from previous step
# ------------------------------------------------------------
rng = np.random.RandomState(0)

def true_function(x):
    return np.sin(2.2 * np.pi * x / 2) * np.exp(-x / 3) + 0.4 * x

X_train = np.array([[0.2], [1.0], [1.8]])
y_train = true_function(X_train).ravel() + 0.05 * rng.randn(len(X_train))

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.6)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gp.fit(X_train, y_train)

X_plot = np.linspace(0, 2, 400).reshape(-1, 1)
mu_n, std_n = gp.predict(X_plot, return_std=True)

x_next = np.array([[1.3]])
mu_next, std_next = gp.predict(x_next, return_std=True)
sigma_next = std_next[0]

# Same three Z_y values
Z_samples = np.array([-1.0, 0.0, 1.0])
colors = ['royalblue', 'gray', 'darkorange']

# Cross-covariance and normalised term
def kernel_matrix(XA, XB): return gp.kernel_(XA, XB)
K = kernel_matrix(X_train, X_train) + gp.alpha * np.eye(len(X_train))
K_inv = np.linalg.inv(K)

def posterior_cov(XA, XB):
    K_XA_XB = kernel_matrix(XA, XB)
    K_XA_Xt = kernel_matrix(XA, X_train)
    K_Xt_XB = kernel_matrix(X_train, XB)
    return K_XA_XB - K_XA_Xt @ K_inv @ K_Xt_XB

cov_X_xnext = posterior_cov(X_plot, x_next).ravel()
tilde_sigma = cov_X_xnext / sigma_next
mu_scenarios = [mu_n + tilde_sigma * z for z in Z_samples]
x_star_idx = [np.argmax(mu_scenarios[i]) for i in range(len(Z_samples))]
x_stars = [X_plot[idx, 0] for idx in x_star_idx]
y_stars = [mu_scenarios[i][idx] for i, idx in enumerate(x_star_idx)]

# ------------------------------------------------------------
# 2) Build linear functions in Z_y for each x*_j
# ------------------------------------------------------------
Z_y = np.linspace(-2.5, 2.5, 200)

# ------------------------------------------------------------
# 2) Build linear functions in Z_y for each x*_j
# ------------------------------------------------------------
Z_y = np.linspace(-2.5, 2.5, 200)

def tilde_sigma_at(x_star):
    cov_xstar_xnext = posterior_cov(np.array([[x_star]]), x_next)[0, 0]
    return cov_xstar_xnext / sigma_next

# ✅ FIX: gp.predict() returns 1D array → index with [0]
mu_star = [gp.predict(np.array([[x]]))[0] for x in x_stars]
tilde_sigma_star = [tilde_sigma_at(x) for x in x_stars]
lines = [mu_star[i] + tilde_sigma_star[i] * Z_y for i in range(3)]
LZ = np.maximum.reduce(lines)  # piecewise-linear envelope


# ------------------------------------------------------------
# 3) Create frames
# ------------------------------------------------------------
outdir = "animation/kg_discrete_L"
os.makedirs(outdir, exist_ok=True)

def make_frame(step):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: GP reminder with discrete set ---
    ax_left.plot(X_plot, mu_n, color='black', lw=1.5)
    ax_left.fill_between(X_plot.ravel(),
                         mu_n - 1.96*std_n, mu_n + 1.96*std_n,
                         color='pink', alpha=0.3)
    ax_left.scatter(X_train, y_train, color='k', s=30)
    for i, x in enumerate(x_stars):
        ax_left.axvline(x, color=colors[i], lw=2, linestyle='--')
        ax_left.text(x-0.05, 2.1, fr"$x^*_{{{i+1}}}$",
                     color=colors[i], fontsize=10, ha='right')
    ax_left.axvline(x_next, color='black', lw=3)
    ax_left.text(1.3, 2.3, r"$x^{n+1}$", ha='center', fontsize=10)
    ax_left.set_xlim(0, 2)
    ax_left.set_ylim(-1.5, 2.5)
    ax_left.set_title("(a) Discrete set $\\mathcal{X}_d$")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("f(x)")

    # --- Right: lines in Z_y space ---
    ax_right.set_xlim(-2.5, 2.5)
    ax_right.set_ylim(min(map(np.min, lines)) - 0.2,
                      max(map(np.max, lines)) + 0.2)
    ax_right.set_xlabel("$Z_y$")
    ax_right.set_ylabel("$\\mu^{n+1}_y(x_j^*)$")
    ax_right.set_title("(b) Linear functions and envelope $L(Z_y)$")

    if step == 0:
        ax_right.text(-2.2, np.min(LZ) + 0.2,
                      "Each $x_j^*$ defines a line in $Z_y$",
                      fontsize=9, color="gray")
    else:
        for i in range(min(step, 3)):
            ax_right.plot(Z_y, lines[i], color=colors[i],
                          lw=1.5, label=fr"$x^*_{{{i+1}}}$")
        if step >= 4:
            ax_right.plot(Z_y, LZ, color='red', lw=2.5,
                          label=r"$L(Z_y)=\max_j \mu^{n+1}_y(x_j^*)$")
        ax_right.legend(loc="lower right")

    plt.tight_layout()
    frame_path = os.path.join(outdir, f"frame_{step:02d}.jpg")
    plt.savefig(frame_path, dpi=200)
    plt.close(fig)
    print(f"Saved {frame_path}")

# Frames: 0 → intro, 1-3 add lines, 4 add envelope
for step in range(0, 5):
    make_frame(step)

print(f"✅ Saved 5 frames under '{outdir}/'")
