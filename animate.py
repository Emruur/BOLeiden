import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# --- True underlying function (smooth + asymmetric) ---
def true_function(x):
    return np.sin(2.5 * np.pi * x / 2) * np.exp(-x / 2) + 0.4 * x

# --- 3 training points, spread out ---
X_train = np.array([[0.2], [1.0], [1.8]])
y_train = true_function(X_train).ravel() + 0.05 * np.random.randn(3)

# --- GP setup: broader kernel + moderate noise for larger variance ---
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.6, length_scale_bounds=(1e-2, 3.0))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gp.fit(X_train, y_train)

# --- Prediction grid ---
X = np.linspace(0, 2.0, 400).reshape(-1, 1)
y_pred, sigma = gp.predict(X, return_std=True)
old_max_idx = np.argmax(y_pred)
old_max_x, old_max_y = X[old_max_idx, 0], y_pred[old_max_idx]

# --- Candidate point ---
x_new = np.array([[1.3]])
y_new_mean, y_new_std = gp.predict(x_new, return_std=True)

# Deliberately sample +1.5 std above mean for visible change
y_new_sample = y_new_mean[0] + 1.5 * y_new_std[0]

# --- Update GP with this "surprising" sample ---
X_upd = np.vstack((X_train, x_new))
y_upd = np.append(y_train, y_new_sample)
gp_upd = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gp_upd.fit(X_upd, y_upd)
y_pred_new, sigma_new = gp_upd.predict(X, return_std=True)
new_max_idx = np.argmax(y_pred_new)
new_max_x, new_max_y = X[new_max_idx, 0], y_pred_new[new_max_idx]

# --- Knowledge Gradient value (posterior mean improvement) ---
KG_value = new_max_y - old_max_y

# --- Create folder for frames ---
os.makedirs("animation", exist_ok=True)

# --- Helper: consistent axis setup ---
def setup_axes(ax):
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-1.5, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.axvline(1.3, color='gray', linestyle='--', lw=1)
    ax.text(1.3, 1.8, "$x^{n+1}$", ha='center', va='bottom', color='gray', fontsize=10)
    return ax

# --- FRAME 1: Before sampling ---
fig, ax = plt.subplots(figsize=(8,4))
ax = setup_axes(ax)
ax.plot(X, y_pred, 'b', label="GP mean")
ax.fill_between(X.ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma, alpha=0.25, color='blue')
ax.scatter(X_train, y_train, color='k', label="Observations")
ax.axhline(old_max_y, color='red', linestyle='--', lw=1, label="Current GP max")
ax.errorbar(x_new, y_new_mean, yerr=1.96*y_new_std, fmt='o', color='orange', capsize=6, label="Predictive var at $x^{n+1}$")
ax.set_title("1️⃣ Before sampling at $x^{n+1}=1.3$")
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("animation/frame_1.jpg", dpi=200)
plt.close()

# --- FRAME 2: Realization (sample above mean) ---
fig, ax = plt.subplots(figsize=(8,4))
ax = setup_axes(ax)
ax.plot(X, y_pred, 'b', label="GP mean")
ax.fill_between(X.ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma, alpha=0.25, color='blue')
ax.scatter(X_train, y_train, color='k', label="Observations")
ax.axhline(old_max_y, color='red', linestyle='--', lw=1, label="Old GP max")
ax.scatter(x_new, y_new_sample, color='orange', edgecolors='k', s=100, label=r"Sampled $y^{n+1} = \mu + 1.5\sigma$")
ax.set_title(r"2️⃣ Realization: $y^{n+1} = \mu(x^{n+1}) + 1.5\sigma(x^{n+1})$")
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("animation/frame_2.jpg", dpi=200)
plt.close()

# --- FRAME 3: Updated GP & Knowledge Gradient ---
fig, ax = plt.subplots(figsize=(8,4))
ax = setup_axes(ax)
ax.plot(X, y_pred_new, 'b', label="Updated GP mean")
ax.fill_between(X.ravel(), y_pred_new - 1.96*sigma_new, y_pred_new + 1.96*sigma_new, alpha=0.25, color='blue')
ax.scatter(X_upd, y_upd, color='k', label="Updated observations")
ax.axhline(old_max_y, color='red', linestyle='--', lw=1, label="Old GP max")
ax.axhline(new_max_y, color='green', linestyle='--', lw=1, label="New GP max")
ax.text(1.85, (old_max_y + new_max_y)/2, f"KG(1.3) = {KG_value:.2f}", color='darkgreen',
        rotation=90, va='center', ha='left', fontsize=10)
ax.set_title("3️⃣ After update — new max and Knowledge Gradient")
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("animation/frame_3.jpg", dpi=200)
plt.close()

print("✅ Frames saved: animation/frame_1.jpg, frame_2.jpg, frame_3.jpg")
