import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# -----------------------------
# 1) GP setup (static posterior)
# -----------------------------
rng = np.random.RandomState(0)

def true_function(x):
    return np.sin(2.2 * np.pi * x / 2) * np.exp(-x / 3) + 0.4 * x

# Training data
X_train = np.array([[0.2], [1.0], [1.8]])
y_train = true_function(X_train).ravel() + 0.05 * rng.randn(len(X_train))

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.6, length_scale_bounds=(1e-2, 3.0))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gp.fit(X_train, y_train)

# Domain for plotting GP
X_plot = np.linspace(0.0, 2.0, 400).reshape(-1, 1)
mu_plot, std_plot = gp.predict(X_plot, return_std=True)

# Candidate point (slightly offset for visibility)
x_next = np.array([[1.32]])
mu_next, std_next = gp.predict(x_next, return_std=True)

# -----------------------------
# 2) Discretized domain + right panel setup
# -----------------------------
X_disc = np.arange(0.0, 2.01, 0.25).reshape(-1, 1)

def kernel_matrix(XA, XB):
    return gp.kernel_(XA, XB)

K = kernel_matrix(X_train, X_train) + gp.alpha * np.eye(len(X_train))
K_inv = np.linalg.inv(K)

def posterior_cov(XA, XB):
    K_XA_XB = kernel_matrix(XA, XB)
    K_XA_Xt = kernel_matrix(XA, X_train)
    K_Xt_XB = kernel_matrix(X_train, XB)
    return K_XA_XB - K_XA_Xt @ K_inv @ K_Xt_XB

cov_Xdisc_xnext = posterior_cov(X_disc, x_next).ravel()
sigma_xnext = std_next[0]
mu_Xdisc, _ = gp.predict(X_disc, return_std=True)

Z_y = np.linspace(-2.5, 2.5, 200)
lines = [mu_Xdisc[i] + (cov_Xdisc_xnext[i] / sigma_xnext) * Z_y for i in range(len(X_disc))]
max_curve = np.max(lines, axis=0)

# -----------------------------
# 3) Create frames
# -----------------------------
outdir = "animation_12/kg_frames"
os.makedirs(outdir, exist_ok=True)

def make_frame(step):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left panel: GP ---
    ax_left.plot(X_plot, mu_plot, color="black", lw=1.5)
    ax_left.fill_between(X_plot.ravel(), mu_plot - 1.96 * std_plot, mu_plot + 1.96 * std_plot,
                         color="pink", alpha=0.4)
    ax_left.scatter(X_train, y_train, color="k", s=30)
    ax_left.set_xlim(0, 2)
    ax_left.set_ylim(-1.5, 2.5)
    ax_left.set_title("(a) GP over domain $\\mathbf{X}$")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("f(x)")

    # --- Discretization bars (dotted) ---
    for i, x in enumerate(X_disc.ravel()):
        color = "gray"
        lw = 1.5
        ls = (0, (3, 3))
        if i < step and step > 0:
            color = "green"
        ax_left.axvline(x, color=color, lw=lw, linestyle=ls, alpha=0.9)

    # --- Candidate x^{n+1}: thinner bold black line ---
    ax_left.axvline(x_next, color="black", lw=3, linestyle='-', alpha=1.0)
    ax_left.text(float(x_next), 2.2, "$x^{n+1}$", ha='center', va='bottom', color='black', fontsize=10)

    # --- Right panel ---
    ax_right.set_xlim(-2.5, 2.5)
    ax_right.set_ylim(np.min(lines) - 0.2, np.max(lines) + 0.2)
    ax_right.set_title("(b) $\\mu^{n+1}_i(x)$ vs $Z_y$")
    ax_right.set_xlabel("$Z_y$")
    ax_right.set_ylabel("$\\mu^{n+1}_i(x)$")

    # Plot progress of lines
    if step == 0:
        ax_right.text(-2.2, np.min(lines) + 0.2, "Domain discretization →", fontsize=9, color="gray")
    else:
        for i in range(step):
            ax_right.plot(Z_y, lines[i], color="gray", alpha=0.6, lw=1)

        # Add reparameterisation equation + arrow in second frame
        if step == 1:
            # Pick a representative line to annotate (the last one drawn)
            example_i = 1
            line_y = lines[example_i]
            # Coordinates for arrow
            arrow_x = 1.5
            arrow_y = np.interp(arrow_x, Z_y, line_y)
            # Draw arrow pointing to the line
            ax_right.annotate(
                r"$\mu_y^{n+1}(x)= \mu_y^{n}(x) \;+\; \tilde{\sigma}_y^{\,n}(x, x^{n+1}) \, Z_y$",
                xy=(arrow_x, arrow_y),
                xytext=(0.5, arrow_y + 0.5),
                fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                ha="left"
            )

        # Add final red envelope at the last frame
        if step == len(X_disc):
            ax_right.plot(Z_y, max_curve, color="red", lw=2.5, label=r"$\max_x \mu^{n+1}_i(x)$")
            ax_right.legend(loc="upper left")

    plt.tight_layout()
    frame_path = os.path.join(outdir, f"frame_{step:02d}.jpg")
    plt.savefig(frame_path, dpi=200)
    plt.close(fig)
    print(f"Saved {frame_path}")

# Generate all frames
make_frame(0)
for i in range(1, len(X_disc) + 1):
    make_frame(i)

print(f"✅ {len(X_disc)+1} frames saved under '{outdir}/'")
