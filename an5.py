import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# --- Step 1: Training data ---
X_train = np.array([[-4.0], [0.0], [3.0]])  # 3 data points
y_train = np.sin(X_train).ravel()

# --- Step 2: Fit GP ---
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
gp.fit(X_train, y_train)

# --- Step 3: Prediction grid ---
X = np.linspace(-6, 6, 400).reshape(-1, 1)
y_mean, y_std = gp.predict(X, return_std=True)

# --- Step 4: Choose high-variance point near center ---
center_region = np.logical_and(X.ravel() > 1.0, X.ravel() < 2.0)
x_high_var_idx = np.argmax(y_std[center_region])
x_high_var = X[center_region][x_high_var_idx, 0]
mean_high_var = y_mean[center_region][x_high_var_idx]
std_high_var = y_std[center_region][x_high_var_idx]

# --- Step 5: Setup figure ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-6, 6)
ax.set_ylim(-3, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("GP Sausage Plot with $y^{n+1}$ Sample")

# GP mean and uncertainty band
ax.plot(X, y_mean, "b", lw=2)
ax.fill_between(X.ravel(), y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.2, color="blue")

# Training points (smaller)
ax.scatter(X_train, y_train, c="r", s=40, zorder=10)

# Vertical line at x^{n+1}
vline = ax.axvline(x_high_var, color="gray", linestyle="dotted", linewidth=1.5)
ax.text(x_high_var + 0.1, -2.5, r"$x^{n+1}$", fontsize=14, color="gray")

# Oscillating point (black) + label
(sample_point,) = ax.plot([], [], "o", color="black", ms=12)
text_label = ax.text(0, 0, r"$y^{n+1}$", fontsize=14, color="black", fontweight="bold")

# --- Step 6: Animation ---
def init():
    sample_point.set_data([], [])
    text_label.set_position((0, 0))
    return sample_point, text_label

def animate(i):
    # Oscillate across full ±2σ range
    y_sample = mean_high_var + 2 * std_high_var * np.sin(i * 0.2)
    sample_point.set_data([x_high_var], [y_sample])
    text_label.set_position((x_high_var + 0.3, y_sample + 0.1))
    return sample_point, text_label

# --- Step 7: Animate and save ---
ani = FuncAnimation(fig, animate, init_func=init, frames=100, interval=80, blit=True)
ani.save("gp_sausage.gif", writer=PillowWriter(fps=15))

print("✅ Animation saved as gp_sausage.gif")
