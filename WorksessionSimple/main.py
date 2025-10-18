import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# ==============================================================
# 1. Toy Objective and Constraint
# ==============================================================

class ToyFunction:
    """A complex, highly multi-modal 1D deterministic test function."""
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return (
            0.6 * np.sin(2 * np.pi * x)
            + 0.3 * np.sin(8 * np.pi * x + 0.3)
            + 0.25 * np.cos(14 * np.pi * (x + 0.2))
            + 0.2 * np.sin(30 * np.pi * x ** 1.2 + 0.5)
            + 0.15 * np.cos(50 * np.pi * (x ** 1.1 + 0.1))
            + 0.45 * np.exp(-60 * (x - 0.22) ** 2)
            - 0.4 * np.exp(-80 * (x - 0.78) ** 2)
            + 0.25 * np.exp(-200 * (x - 0.55) ** 4)
            - 0.15 * np.exp(-150 * (x - 0.9) ** 6)
            + 0.1 * x ** 2 - 0.05 * x ** 3
        )


class ToyConstraint:
    """A smooth nonlinear constraint: feasible if c(x) ≤ 0."""
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return 0.3 * np.sin(6 * np.pi * (x - 0.2)) + 0.1 * np.cos(3 * np.pi * (x - 0.2)) - 0.05


# ==============================================================
# 2. Constrained Knowledge Gradient (Single Constraint)
# ==============================================================

class ConstrainedKnowledgeGradient:
    """Constrained Knowledge Gradient acquisition function for one constraint."""

    def __init__(self, gp_obj, gp_con, M=10, sigma_e2_obj=0.0, sigma_e2_con=0.0, rng=None):
        self.gp_obj = gp_obj
        self.gp_con = gp_con
        self.M = M
        self.sigma_e2_obj = sigma_e2_obj
        self.sigma_e2_con = sigma_e2_con
        self.rng = np.random.RandomState(2) if rng is None else rng

    # --- Gaussian Process utilities ---
    def _posterior_mu_std(self, gp, X):
        mu, std = gp.predict(X, return_std=True)
        return mu.ravel(), std.ravel()

    def _tilde_sigma(self, gp, X, xc, sigma_e2):
        """Compute the correlation vector between x' and the new point xc."""
        X_all = np.vstack([X, xc.reshape(1, -1)])
        _, Cov = gp.predict(X_all, return_cov=True)
        k_xprime_xc = Cov[:-1, -1]
        var_xc = Cov[-1, -1]
        denom = np.sqrt(np.maximum(var_xc + sigma_e2, 1e-12))
        return k_xprime_xc / denom

    # --- Probability of Feasibility (single constraint) ---
    def probability_of_feasibility(self, mu_c, std_c):
        return norm.cdf(-mu_c / (std_c + 1e-12))

    # --- Core acquisition computation ---
    def compute(self, X_grid, candidates):
        """
        Compute cKG(x_c) = E[max_x' μ_y^{n+1}(x') * PF^{n+1}(x')] - max_x' μ_y^n(x') * PF^n(x')
        """
        X_grid = np.atleast_2d(X_grid)
        candidates = np.atleast_2d(candidates)

        # 1️⃣ Current posteriors
        mu_y, _ = self._posterior_mu_std(self.gp_obj, X_grid)
        mu_c, std_c = self._posterior_mu_std(self.gp_con, X_grid)

        # 2️⃣ Current feasibility-weighted objective
        PF_n = self.probability_of_feasibility(mu_c, std_c)
        baseline = np.max(mu_y * PF_n)

        # 3️⃣ Monte Carlo samples for objective and constraint outcomes
        Zy = self.rng.randn(self.M)
        Zc = self.rng.randn(self.M)
        ckg = np.zeros(len(candidates))

        # 4️⃣ Evaluate each candidate xc
        for i, xc in enumerate(candidates):
            # Influence (reparametrized variance)
            ## Implement start
            tilde_y = self._tilde_sigma(self.gp_obj, X_grid, xc, self.sigma_e2_obj)
            tilde_c = self._tilde_sigma(self.gp_con, X_grid, xc, self.sigma_e2_con)

            # Updated constraint variance after observing xc
            std_c_next = np.sqrt(np.maximum(std_c**2 - tilde_c**2, 1e-12))

            vals = []
            for m in range(self.M):
                # Hypothetical posterior means after seeing one new sample
                mu_y_next = mu_y + tilde_y * Zy[m]
                mu_c_next = mu_c + tilde_c * Zc[m]
                PF_next = self.probability_of_feasibility(mu_c_next, std_c_next)
                vals.append(np.max(mu_y_next * PF_next))
                
            # Implement end

            ckg[i] = np.mean(vals) - baseline

        return ckg


# ==============================================================
# 3. Bayesian Optimizer with Visualization
# ==============================================================

class BayesianOptimizer:
    def __init__(self, func, constraint, run_id=4, n_steps=8, init_points=5, m_mc=10):
        self.func = func
        self.constraint = constraint
        self.n_steps = n_steps
        self.m_mc = m_mc
        self.grid_n = 200
        self.candidates = np.linspace(0, 1, 101).reshape(-1, 1)
        self.X_grid = np.linspace(0, 1, self.grid_n).reshape(-1, 1)

        # Initial training data
        self.X_train = np.linspace(0, 1, init_points).reshape(-1, 1)
        self.y_train = func(self.X_train).ravel()
        self.yc_train = constraint(self.X_train).ravel()

        self.run_dir = Path(f"runs/ckg_single/run{run_id:03d}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress = []  # record best feasible value per step

    def _fit_gp(self, X, y):
        kernel = C(1.0, (1e-3, 10)) * RBF(length_scale=0.3, length_scale_bounds=(1e-2, 10))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, random_state=0, n_restarts_optimizer=5)
        gp.fit(X, y)
        return gp

    def _visualize(self, gpr_obj, gpr_con, acq_values, x_next, step):
        """Plot function, constraint, GP posteriors, and acquisition."""
        X_plot = np.linspace(0, 1, self.grid_n).reshape(-1, 1)
        mu, std = gpr_obj.predict(X_plot, return_std=True)
        true_y = self.func(X_plot)

        mu_c, std_c = gpr_con.predict(X_plot, return_std=True)
        true_c = self.constraint(X_plot)
        feasible_mask = true_c <= 0

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # (1) True function with feasible region
        axes[0, 0].plot(X_plot, true_y, "k", label="True f(x)")
        axes[0, 0].fill_between(X_plot.ravel(), np.min(true_y), np.max(true_y),
                                where=feasible_mask.ravel(), color="green", alpha=0.15, label="Feasible region")
        axes[0, 0].scatter(self.X_train, self.y_train, color="tab:blue", marker="x")
        axes[0, 0].axvline(x_next, color="red", linestyle="--", lw=1)
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_title(f"True f(x) with Feasible Region (Step {step})")
        axes[0, 0].legend()

        # (2) Objective GP Posterior
        axes[0, 1].plot(X_plot, mu, color="tab:orange", label="GP mean")
        axes[0, 1].fill_between(X_plot.ravel(), mu - 2 * std, mu + 2 * std, color="tab:orange", alpha=0.2)
        axes[0, 1].scatter(self.X_train, self.y_train, color="k", s=15)
        axes[0, 1].axvline(x_next, color="red", linestyle="--")
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_title("Objective GP Posterior ±2σ")

        # (3) Constraint GP Posterior
        axes[1, 0].plot(X_plot, true_c, "k", label="True c(x)")
        axes[1, 0].plot(X_plot, mu_c, color="tab:red", label="GP mean")
        axes[1, 0].fill_between(X_plot.ravel(), mu_c - 2 * std_c, mu_c + 2 * std_c, color="tab:red", alpha=0.2)
        axes[1, 0].axhline(0, color="gray", lw=1, linestyle="--")
        axes[1, 0].scatter(self.X_train, self.yc_train, color="black", s=15)
        axes[1, 0].axvline(x_next, color="red", linestyle="--")
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_title("Constraint GP Posterior ±2σ")
        axes[1, 0].legend()

        # (4) Acquisition function
        axes[1, 1].plot(self.candidates, acq_values, color="tab:green")
        axes[1, 1].axvline(x_next, color="red", linestyle="--", label="max CKG")
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_title("Constrained KG Acquisition")

        plt.tight_layout()
        plt.show()

    # ---------- Final analysis ----------
    def _final_analysis(self):
        X_dense = np.linspace(0, 1, 5000).reshape(-1, 1)
        f_dense = self.func(X_dense)
        c_dense = self.constraint(X_dense)
        feasible_mask = c_dense <= 0
        feasible_y = np.where(feasible_mask, f_dense, -np.inf)
        idx_best = np.argmax(feasible_y)
        x_best_true = X_dense[idx_best, 0]
        f_best_true = feasible_y[idx_best, 0]

        print(f"Theoretical feasible optimum: x*={x_best_true:.3f}, f(x*)={f_best_true:.3f}")

        # Distance to true feasible optimum
        dists = [abs(f_best_true - v) for v in self.progress if not np.isnan(v)]
        steps = np.arange(1, len(dists) + 1)

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(X_dense, f_dense, "k", label="True f(x)")
        ax[0].fill_between(X_dense.ravel(), np.min(f_dense), np.max(f_dense),
                           where=feasible_mask.ravel(), color="green", alpha=0.15, label="Feasible region")
        ax[0].axvline(x_best_true, color="red", linestyle="--", label="Theoretical optimum")
        ax[0].set_xlim(0, 1)
        ax[0].set_title("True Function and Feasible Region")
        ax[0].legend()

        ax[1].plot(steps, dists, "o-", color="tab:blue")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("|f_best_feasible - f*(true)|")
        ax[1].set_title("Distance to Theoretical Feasible Optimum")
        plt.tight_layout()
        plt.show()

    # ---------- Optimization loop ----------
    def run(self):
        for step in range(self.n_steps):
            print(f"--- Step {step+1}/{self.n_steps} ---")

            gp_obj = self._fit_gp(self.X_train, self.y_train)
            gp_con = self._fit_gp(self.X_train, self.yc_train)

            acq = ConstrainedKnowledgeGradient(gp_obj, gp_con, M=self.m_mc)
            acq_values = acq.compute(self.X_grid, self.candidates)

            x_next = self.candidates[np.argmax(acq_values)]
            y_next = self.func(x_next)
            yc_next = self.constraint(x_next)

            feas_mask = np.array(self.yc_train) <= 0
            best_feas = np.max(self.y_train[feas_mask]) if np.any(feas_mask) else np.nan
            self.progress.append(best_feas)

            print(f"x_next={x_next[0]:.3f}, f(x)={y_next[0]:.3f}, c(x)={yc_next[0]:.3f}")

            self._visualize(gp_obj, gp_con, acq_values, x_next, step)

            self.X_train = np.vstack([self.X_train, x_next])
            self.y_train = np.append(self.y_train, y_next)
            self.yc_train = np.append(self.yc_train, yc_next)

        self._final_analysis()
        print("Optimization finished. Plots saved under:", self.run_dir)


# ==============================================================
# 4. Run Example
# ==============================================================

if __name__ == "__main__":
    f = ToyFunction()
    c = ToyConstraint()

    print("Running constrained Bayesian optimization (1D, single constraint)...")
    bo = BayesianOptimizer(f, constraint=c, n_steps=8, init_points=5, m_mc=500)
    bo.run()
