import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ------------------------------------------------------------
# 1) GP setup
# ------------------------------------------------------------
rng = np.random.RandomState(0)

def true_function(x):
    return np.sin(2.2 * np.pi * x / 2) * np.exp(-x / 3) + 0.4 * x

# Training data
X_train = np.array([[0.2], [1.0], [1.8]])
y_train = true_function(X_train).ravel() + 0.05 * rng.randn(len(X_train))

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.6, length_scale_bounds=(1e-2, 3.0))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gp.fit(X_train, y_train)

X_plot = np.linspace(0, 2, 400).reshape(-1, 1)
mu_n, std_n = gp.predict(X_plot, return_std=True)

# Candidate x^{n+1}
x_next = np.array([[1.3]])
mu_next, std_next = gp.predict(x_next, return_std=True)
sigma_next = std_next[0]

# ------------------------------------------------------------
# 2) Reparameterization-based update for different Z_y values
# ------------------------------------------------------------
def kernel_matrix(XA, XB):
    return gp.kernel_(XA, XB)

K = kernel_matrix(X_train, X_train) + gp.alpha * np.eye(len(X_train))
K_inv = np.linalg.inv(K)

def posterior_cov(XA, XB):
    K_XA_XB = kernel_matrix(XA, XB)
    K_XA_Xt = kernel_matrix(XA, X_train)
    K_Xt_XB = kernel_matrix(X_train, XB)
    return K_XA_XB - K_XA_Xt @ K_inv @ K_Xt_XB

# Compute cross-covariance terms
cov_X_xnext = posterior_cov(X_plot, x_next).ravel()
tilde_sigma = cov_X_xnext / sigma_next

# Choose quantiles for Z_y
Z_samples = np.array([-1.0, 0.0, 1.0])
colors = ['royalblue', 'gray', 'darkorange']

# Compute scenario means
mu_scenarios = [mu_n + tilde_sigma * z for z in Z_samples]

# Find maxima for each scenario
x_star_idx = [np.argmax(mu_scenarios[i]) for i in range(len(Z_samples))]
x_stars = [X_plot[idx, 0] for idx in x_star_idx]
y_stars = [mu_scenarios[i][idx] for i, idx in enumerate(x_star_idx)]

# Compute y^{n+1} realizations (at x_next)
y_next_values = [mu_next + sigma_next * z for z in Z_samples]

# ------------------------------------------------------------
# 3) Create animation frames
# ------------------------------------------------------------
outdir = "animation/kg_highvalue_Xd"
os.makedirs(outdir, exist_ok=True)

def make_frame(step):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(X_plot, mu_n, color='black', lw=1.5, label='Current GP mean')
    ax.fill_between(X_plot.ravel(), mu_n - 1.96 * std_n, mu_n + 1.96 * std_n,
                    color='pink', alpha=0.3)
    ax.scatter(X_train, y_train, color='k', s=30)
    ax.axvline(x_next, color='black', lw=3, label=r"$x^{n+1}$")
    ax.set_xlim(0, 2)
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Obtaining High-Value Points $\\mathcal{X}_d$")

    # Step 0: show all faint scenario lines
    if step == 0:
        for i in range(3):
            ax.plot(X_plot, mu_scenarios[i], color=colors[i], alpha=0.3, lw=1)
        ax.text(0.05, 2.2, "Hypothetical posteriors for $Z_y = \\{-1, 0, 1\\}$", fontsize=9.5)
    # Steps 1–3: highlight each scenario one by one
    elif 1 <= step <= 3:
        i = step - 1
        for j in range(3):
            ax.plot(X_plot, mu_scenarios[j], color=colors[j], alpha=0.15, lw=1)
        ax.plot(X_plot, mu_scenarios[i], color=colors[i], lw=2)
        ax.scatter(x_stars[i], y_stars[i], s=100, color=colors[i], edgecolors='k', zorder=5)
        ax.text(x_stars[i]-0.05, y_stars[i]+0.15, fr"$x^*_{{{i+1}}}$", color=colors[i], fontsize=11, ha='right')
        # show y^{n+1} point
        ax.scatter(x_next, y_next_values[i], s=80, color=colors[i], edgecolors='k', zorder=5)
        ax.text(float(x_next)+0.05, float(y_next_values[i]), r"$y^{n+1}$", color=colors[i], fontsize=9)
        ax.text(0.05, 2.2, fr"Scenario for $Z_y={Z_samples[i]:.0f}$", fontsize=10, color=colors[i])
    # Step 4: show all scenarios and their maxima (final collection)
    else:
        for i in range(3):
            ax.plot(X_plot, mu_scenarios[i], color=colors[i], lw=1.5)
            ax.scatter(x_stars[i], y_stars[i], s=100, color=colors[i], edgecolors='k', zorder=5)
            # labels shifted left
            ax.text(x_stars[i]-0.05, y_stars[i]+0.15, fr"$x^*_{{{i+1}}}$", color=colors[i], fontsize=10, ha='right')
            # Add vertical bars showing discrete set collection
            ax.axvline(x_stars[i], color=colors[i], lw=3, linestyle='--', alpha=0.7)
            # show y^{n+1} point (same color as line)
            ax.scatter(x_next, y_next_values[i], s=80, color=colors[i], edgecolors='k', zorder=5)
        ax.text(0.05, 2.2, r"Discrete set $\mathcal{X}_d = \{x_1^*, x_2^*, x_3^*\}$", fontsize=10, color='k')

    ax.legend(loc='lower left')
    plt.tight_layout()
    frame_path = os.path.join(outdir, f"frame_{step:02d}.jpg")
    plt.savefig(frame_path, dpi=200)
    plt.close(fig)
    print(f"Saved {frame_path}")

# Generate frames
for step in range(0, 5):
    make_frame(step)

print(f"✅ Saved 5 frames under '{outdir}/'")
