import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List, Callable, Tuple, Dict
import matplotlib.colors as mcolors
import json
import os
from scipy.stats import norm
import numpy as np
from numpy.linalg import solve


# ===========================================================
# 1. RBF Kernel
# ===========================================================
class RBFKernel:
    """Squared exponential (RBF) kernel."""
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dists = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1)
        return self.variance * np.exp(-0.5 * dists / self.lengthscale**2)


# ===========================================================
# 2. Gaussian Process model
# ===========================================================
class GPModel:
    """Lightweight GP regression model using exact inference."""
    def __init__(self, kernel: RBFKernel, noise: float = 1e-6):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.L = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data."""
        self.X_train, self.y_train = X, y
        K = self.kernel(X, X) + self.noise * np.eye(len(X))
        self.L = np.linalg.cholesky(K)

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and variance at X_star."""
        if self.X_train is None:
            # Handle cases where GP hasn't been fitted yet, return prior mean/variance
            return np.zeros(len(X_star)), np.full(len(X_star), self.kernel.variance + self.noise)

        K_x = self.kernel(self.X_train, X_star)
        K_ss = self.kernel(X_star, X_star)
        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))
        mu = K_x.T @ alpha
        v = np.linalg.solve(self.L, K_x)
        cov = K_ss - v.T @ v
        return mu.ravel(), np.clip(np.diag(cov), 1e-12, None)


# ===========================================================
# 3. Constrained Knowledge Gradient (Algorithm 1)
# ===========================================================
def _post_cross_terms(gp, X: np.ndarray, x_next: np.ndarray):
        """
        Compute posterior cross-covariance k^n(X, x_next) and k^n(x_next,x_next)
        using the GP's cached Cholesky factor gp.L (of K + noise I).
        """
        x_next = x_next.reshape(1, -1)
        # prior blocks
        K_X_xn1      = gp.kernel(X, x_next)                   # (N, 1)
        K_X_Xtr      = gp.kernel(X, gp.X_train)               # (N, n)
        K_Xtr_xn1    = gp.kernel(gp.X_train, x_next)          # (n, 1)
        k_xn1_xn1_pr = gp.kernel(x_next, x_next).item()       # scalar

        # (K + σ^2 I)^{-1} * k(X_tr, x_next) via Cholesky
        tmp = solve(gp.L, K_Xtr_xn1)               # (n,1)
        alpha = solve(gp.L.T, tmp)                 # (n,1)

        # Posterior cross-covariances
        k_post_X_xn1   = (K_X_xn1 - K_X_Xtr @ alpha).ravel()       # (N,)
        k_post_xn1_xn1 = float(k_xn1_xn1_pr - (tmp.T @ tmp))       # scalar

        return k_post_X_xn1, k_post_xn1_xn1


def _expected_max_of_lines(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return E[ max_i (a_i + b_i Z) ] for Z ~ N(0,1) using the
    high-value linear envelope (upper hull with monotone slopes).
    """
    if len(a) == 0:
        return 0.0

    # Sort by slope; drop dominated equal-slope lines
    order = np.argsort(b)
    a, b = a[order], b[order]

    hull_idx = []
    breaks = []  # z where a new line becomes active; len == len(hull_idx)

    for i in range(len(a)):
        # remove dominated equal-slope
        while hull_idx and np.isclose(b[i], b[hull_idx[-1]], atol=1e-12):
            if a[i] <= a[hull_idx[-1]]:
                # current line never better
                break
            else:
                # replace previous
                hull_idx.pop()
                breaks.pop()
        else:
            # main upper-hull loop
            while hull_idx:
                j = hull_idx[-1]
                # intersection z* where new i overtakes j
                z_star = (a[j] - a[i]) / (b[i] - b[j])
                if len(hull_idx) == 1:
                    last_break = -np.inf
                else:
                    last_break = breaks[-1]
                if z_star <= last_break + 1e-14:
                    # j can never be optimal
                    hull_idx.pop()
                    breaks.pop()
                else:
                    break
            else:
                z_star = -np.inf

            # append i if it survived
            if (not hull_idx) or (not np.isclose(b[i], b[hull_idx[-1]], atol=1e-12) or (a[i] > a[hull_idx[-1]])):
                hull_idx.append(i)
                breaks.append(z_star)

    # integrate each active segment
    breaks.append(np.inf)  # right boundary
    E = 0.0
    for seg in range(len(hull_idx)):
        i = hull_idx[seg]
        L, R = breaks[seg], breaks[seg+1]
        # Contribution: ∫_{L}^{R} (a_i + b_i z) φ(z) dz
        E += a[i] * (norm.cdf(R) - norm.cdf(L)) + b[i] * (norm.pdf(L) - norm.pdf(R))
    return float(E)
    
class ConstrainedKnowledgeGradient:
    """
    Implementation of Algorithm 1: cKG computation (Ungredda & Branke, 2024)
    """

    def __init__(
        self,
        gp_objective: GPModel,
        gp_constraints: List[GPModel],
        candidate_X: np.ndarray,
        nc: int = 10,
        ny: int = 10,
    ):
        """
        Parameters
        ----------
        gp_objective : GPModel
            Gaussian process model for the objective function.
        gp_constraints : list of GPModel
            Gaussian process models for the constraints.
        candidate_X : ndarray (N, d)
            Discretised search space X_d.
        nc, ny : int
            Monte Carlo discretisation sizes for constraints and objective.
        """
        self.gp_y = gp_objective
        self.gp_constraints = gp_constraints
        self.candidate_X = candidate_X
        self.nc = nc
        self.ny = ny
        self.nz = nc * ny  # Line 0 – total Monte Carlo samples

    # ------------------------------------------------------------
    def __call__(self, x_next: np.ndarray) -> float:
        """Compute cKG(x_next) according to Algorithm 1."""
        # Ensure x_next is 2D for GP.predict
        x_next_2d = x_next.reshape(1, -1)

        # Line 1 – Compute incumbent best feasible xⁿ
        mu_y, var_y = self.gp_y.predict(self.candidate_X)
        pf = self._prob_feasible(self.candidate_X)
        
        # Handle cases where no feasible point is found yet (pf might be all zeros or very small)
        if np.all(pf == 0) or np.all(mu_y * pf == 0):
            # If no feasible point, cKG might need to explore,
            # for simplicity, we can default x_n to a random point or the first candidate
            x_n = self.candidate_X[0] 
        else:
            x_n = self.candidate_X[np.argmax(mu_y * pf)]

        # Line 0 – Initialise discretisation set X_d⁰ = ∅
        X_d = []

        # Lines 2–5 – Compute discretisation points
        X_d = self._compute_discretisation_points(x_next_2d, X_d)

        # Handle case where X_d could be empty if nz is too small or candidates too sparse
        if len(X_d) == 0:
            return 0.0 # No points to evaluate KG_d over

        # Lines 6–8 – Monte Carlo estimation of KG_d
        kg_values = []
        for _ in range(self.nc):
            Z_c = np.random.randn(len(self.gp_constraints))
            kg_m = self._compute_KGd(x_next_2d, X_d, Z_c)
            kg_values.append(kg_m)

        # Line 8 – Monte Carlo average
        ckg_val = np.mean(kg_values)

        # Line 9 – Return final cKG(x_next)
        return float(ckg_val)

    # ------------------------------------------------------------
    def _compute_discretisation_points(self, x_next: np.ndarray, X_d_list: list) -> np.ndarray:
        """
        Algorithm 1 – Lines 2–5 (reparameterized version)
        Compute Monte Carlo discretisation points X_d by iterating n_z times.
        Each iteration samples Z_y, Z_c and maximizes μ_y^{n+1}(x).
        """
        if len(self.candidate_X) == 0:
            return np.array([])

        for j in range(self.nz):
            # Line 3 – Draw Z_yʲ, Z_cʲ ~ N(0, 1)
            Z_yj = np.random.randn()
            Z_cj = np.random.randn(len(self.gp_constraints))

            # Get posterior mean/var at step n
            mu_y, var_y = self.gp_y.predict(self.candidate_X)


            # Compute posterior covariance terms for TODO reparameterization
            gp = self.gp_y
            K_x_xn1 = gp.kernel(gp.X_train, x_next.reshape(1, -1))
            K_X_xn1 = gp.kernel(self.candidate_X, x_next.reshape(1, -1))
            K_xn1_xn1 = gp.kernel(x_next.reshape(1, -1), x_next.reshape(1, -1)).item()

            v = np.linalg.solve(gp.L, K_x_xn1)
            k_post_X_xn1 = (K_X_xn1 - gp.kernel(self.candidate_X, gp.X_train) @ np.linalg.solve(gp.L.T, v)).ravel()
            v_self = np.linalg.solve(gp.L, gp.kernel(gp.X_train, x_next.reshape(1, -1)))
            k_post_xn1_xn1 = K_xn1_xn1 - float(v_self.T @ v_self)

            sigma_tilde_y = k_post_X_xn1 / np.sqrt(k_post_xn1_xn1 + gp.noise)

            # Reparameterized posterior mean (μ_y^{n+1}(x))
            mu_y_updated = mu_y + sigma_tilde_y * Z_yj

            # Probability of feasibility using reparameterized constraints
            pf_next = self._prob_feasible(self.candidate_X, z_c=Z_cj, x_next=x_next)

            # Acquisition value (deterministic mean times feasibility)
            val = mu_y_updated * pf_next

            if np.all(val == 0):
                x_star_j = self.candidate_X[np.random.randint(len(self.candidate_X))]
            else:
                x_star_j = self.candidate_X[np.argmax(val)]

            X_d_list.append(x_star_j)

        return np.unique(np.array(X_d_list), axis=0)

    # ------------------------------------------------------------
    

    # ---------- helpers ----------
    


    # ---------- main: KG_d ----------
    def _compute_KGd(self, x_next: np.ndarray, X_d: np.ndarray, Z_c: np.ndarray) -> float:
        """
        Compute KG_d(x^{n+1}; Z_c) with fixed Z_c:

        KG_d = E_{Z_y} [ max_{x'∈X_d} { (μ_y^n(x') + σ̃_y^n(x',x^{n+1}) Z_y) · PF^{n+1}(x'; x^{n+1}, Z_c) } ]
                - max_{x'∈X_d} μ_y^n(x') · PF^n(x')

        Uses the high-value linear envelope to evaluate the expectation in closed form.
        """
        if X_d.size == 0:
            return 0.0

        # Current penalized best: max μ^n(x) PF^n(x)
        mu_y_Xd, _ = self.gp_y.predict(X_d)
        pf_curr_Xd = self._prob_feasible(X_d)            # PF^n
        best_current = float(np.max(mu_y_Xd * pf_curr_Xd))

        # Reparameterized feasibility at n+1 (depends only on x_next, Z_c)
        pf_next_Xd = self._prob_feasible(X_d, z_c=Z_c, x_next=x_next)  # PF^{n+1}(·; x_next, Z_c)

        # σ̃_y^n(x, x_next)
        k_post_X_xn1, k_post_xn1_xn1 = _post_cross_terms(self.gp_y, X_d, x_next)
        denom = np.sqrt(max(k_post_xn1_xn1 + self.gp_y.noise, 1e-18))
        sigma_tilde = k_post_X_xn1 / denom

        # Lines a_i + b_i Z_y for the envelope
        a = mu_y_Xd * pf_next_Xd
        b = sigma_tilde * pf_next_Xd

        # Expectation of the max over Z_y ~ N(0,1)
        Em = _expected_max_of_lines(a, b)

        return Em - best_current

    # ------------------------------------------------------------
    # ------------------------------------------------------------


    def _prob_feasible(self, X: np.ndarray, z_c: np.ndarray = None, x_next: np.ndarray = None) -> np.ndarray:
        """
        Compute PFⁿ(x) or PFⁿ⁺¹(x; xⁿ⁺¹, Z_c) if z_c and x_next are given.

        Implements the reparameterized update:
            μ_k^{n+1}(x) = μ_k^n(x) + σ̃_k^n(x, xⁿ⁺¹) * Z_k
            k_k^{n+1}(x,x) = k_k^n(x,x) - (σ̃_k^n(x, xⁿ⁺¹))²
        where σ̃_k^n(x, xⁿ⁺¹) = k_k^n(x, xⁿ⁺¹) / sqrt(k_k^n(xⁿ⁺¹, xⁿ⁺¹) + σ²_noise)
        """
        if X.size == 0:
            return np.array([])

        pf_total = np.ones(len(X))

        for k, gp in enumerate(self.gp_constraints):
            mu_x, var_x = gp.predict(X)
            var_x = np.clip(var_x, 1e-12, None)

            if z_c is None or x_next is None:
                # Regular (current) PF^n(x)
                pf_total *= norm.cdf(-mu_x / np.sqrt(var_x))
                continue

            # === Reparameterized PF^{n+1}(x; x_next, Z_c) ===
            # Posterior covariance terms at step n
            K_x_xn1 = gp.kernel(gp.X_train, x_next.reshape(1, -1))
            K_X_xn1 = gp.kernel(X, x_next.reshape(1, -1))

            # k^n(x_next, x_next)
            K_xn1_xn1 = gp.kernel(x_next.reshape(1, -1), x_next.reshape(1, -1)).item()

            # Compute posterior covariance using Cholesky (no need to re-invert)
            v = np.linalg.solve(gp.L, K_x_xn1)
            k_post_X_xn1 = (K_X_xn1 - gp.kernel(X, gp.X_train) @ np.linalg.solve(gp.L.T, v)).ravel()

            v_self = np.linalg.solve(gp.L, gp.kernel(gp.X_train, x_next.reshape(1, -1)))
            k_post_xn1_xn1 = K_xn1_xn1 - float(v_self.T @ v_self)

            # Compute σ̃_k^n(x, x_next)
            sigma_tilde = k_post_X_xn1 / np.sqrt(k_post_xn1_xn1 + gp.noise)

            # Apply reparameterization update
            mu_upd = mu_x + sigma_tilde * z_c[k]
            var_upd = np.clip(var_x - sigma_tilde**2, 1e-12, None)

            # PF^{n+1}(x)
            pf_total *= norm.cdf(-mu_upd / np.sqrt(var_upd))

        return pf_total



class ConstrainedExpectedImprovement:
    """
    Computes the Constrained Expected Improvement (cEI) acquisition function.

    cEI is calculated as the product of the standard Expected Improvement (EI)
    and the Probability of Feasibility (PF).
    
    cEI(x) = EI(x) * PF(x)
    """

    def __init__(
        self,
        gp_objective: GPModel,
        gp_constraints: List[GPModel],
        best_feasible_y: float,
    ):
        """
        Parameters
        ----------
        gp_objective : GPModel
            Gaussian process model for the objective function.
        gp_constraints : list of GPModel
            A list of Gaussian process models for the constraints.
            Each constraint is assumed to be in the form g(x) <= 0.
        best_feasible_y : float
            The best observed objective value among all feasible points found so far.
            This is often denoted as f(x+).
        """
        self.gp_y = gp_objective
        self.gp_constraints = gp_constraints
        self.best_feasible_y = best_feasible_y
        self.eps = 1e-9 # Small constant to prevent division by zero

    # ------------------------------------------------------------
    def __call__(self, x_next: np.ndarray) -> float:
        """
        Compute cEI(x_next) for a single candidate point.

        Parameters
        ----------
        x_next : ndarray (d,) or (1, d)
            The candidate point at which to evaluate cEI.

        Returns
        -------
        float
            The cEI value. A higher value is better.
        """
        # Ensure x_next is a 2D array for GP prediction
        x_next = np.atleast_2d(x_next)

        # 1. Calculate standard Expected Improvement (EI) for the objective
        ei = self._expected_improvement(x_next)

        # 2. Calculate Probability of Feasibility (PF) for the constraints
        pf = self._prob_feasible(x_next)

        # 3. Compute cEI = EI * PF
        cei_value = ei * pf

        return float(cei_value)

    # ------------------------------------------------------------
    def _expected_improvement(self, x: np.ndarray) -> float:
        """Calculates the standard Expected Improvement at point x."""
        # Get posterior mean and variance from the objective GP
        mu_y, var_y = self.gp_y.predict(x)
        
        # Ensure variance is non-negative
        var_y = np.clip(var_y, self.eps, None)
        sigma_y = np.sqrt(var_y)
        
        # Difference between mean prediction and current best
        improvement = mu_y - self.best_feasible_y
        
        # Standardized improvement term Z
        Z = improvement / sigma_y
        
        # EI formula
        ei = improvement * norm.cdf(Z) + sigma_y * norm.pdf(Z)
        
        return ei.item()

    # ------------------------------------------------------------
    def _prob_feasible(self, x: np.ndarray) -> float:
        """Calculates the joint Probability of Feasibility at point x."""
        if not self.gp_constraints:
            # If there are no constraints, feasibility is 100%
            return 1.0

        pf_total = 1.0
        for gp_c in self.gp_constraints:
            # Get posterior mean and variance from the constraint GP
            mu_c, var_c = gp_c.predict(x)
            
            # Ensure variance is non-negative
            var_c = np.clip(var_c, self.eps, None)
            sigma_c = np.sqrt(var_c)

            # Probability that g(x) <= 0 is norm.cdf(-mu_c / sigma_c)
            # This is P(N(mu_c, sigma_c^2) <= 0)
            pf_total *= norm.cdf(-mu_c / sigma_c)

        return pf_total.item()


# ===========================================================
# 4. Bayesian Optimizer
# ===========================================================
class BayesianOptimizer:
    """Bayesian optimization with constrained Knowledge Gradient and final performance plot."""
    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        constraint_fns: List[Callable[[np.ndarray], float]],
        bounds: np.ndarray,
        kernel_params: Dict = None,
        noise: float = 1e-6,
        plot_steps: bool = True,
        grid_density: int = 50,
        base_dir: str = "runs" # Base directory for all AF runs
    ):
        self.obj_fn = objective_fn
        self.cons_fns = constraint_fns
        self.bounds = bounds
        kernel = RBFKernel(**(kernel_params or {}))
        self.obj_model = GPModel(kernel, noise)
        self.cons_models = [GPModel(kernel, noise) for _ in constraint_fns]
        self.X, self.y, self.C = None, None, None
        self.best_vals = []
        self.iterations = []
        self.theoretical_max = None
        self.plot_steps = plot_steps
        self.grid_density = grid_density
        self.base_dir = base_dir
        self.af = None # Placeholder for acquisition function name
        self.run_dir = None # Will be set in initialise

    def _setup_run_directory(self):
        """Creates a unique directory for the current run based on AF."""
        if not self.af:
            raise ValueError("Acquisition function (self.af) must be set before setting up the run directory.")
            
        af_dir = os.path.join(self.base_dir, self.af)
        os.makedirs(af_dir, exist_ok=True)
        
        # Find the highest run number in the specific AF directory
        run_numbers = []
        for d in os.listdir(af_dir):
            if d.startswith("run") and os.path.isdir(os.path.join(af_dir, d)):
                try:
                    # Expects format 'runX'
                    run_numbers.append(int(d[3:])) 
                except ValueError:
                    continue
        
        next_run_num = max(run_numbers) + 1 if run_numbers else 1
        current_run_dir = os.path.join(af_dir, f"run{next_run_num}")
        os.makedirs(current_run_dir)
        print(f"Saving run data to: {current_run_dir}")
        return current_run_dir

    def initialise(self, n_init: int = 5, af: str = "ckg"):
        """Initial random sampling of the objective and constraints and sets up the run directory."""
        
        # Set the acquisition function and set up the directory *before* first plotting
        self.af = af
        self.run_dir = self._setup_run_directory()
        
        X = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                              size=(n_init, self.bounds.shape[0]))
        y = np.array([self.obj_fn(x) for x in X])
        C = np.array([[fn(x) for fn in self.cons_fns] for x in X])
        self.X, self.y, self.C = X, y, C
        self.obj_model.fit(X, y)
        for k, gp in enumerate(self.cons_models):
            gp.fit(X, C[:, k])
        
        self.theoretical_max = self._compute_true_max()
        
        if self.plot_steps:
            print("Initialisation complete. Plotting initial state.")
            self._plot_acquisition_and_gps(
                title="Initial State (Iteration 0)",
                acq_candidate_X=None,
                acq_vals=None,
                next_x_to_sample=None,
                filename="iter0.jpg"
            )

    def _compute_true_max(self, n_grid: int = 2000):
        """Compute theoretical constrained maximum via dense grid sampling."""
        grid = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                 size=(n_grid, self.bounds.shape[0]))
        feas_mask = np.ones(len(grid), dtype=bool)
        for fn in self.cons_fns:
            feas_mask &= (np.array([fn(x) for x in grid]) <= 0)
        if not np.any(feas_mask):
            return None
        feasible_points = grid[feas_mask]
        values = np.array([self.obj_fn(x) for x in feasible_points])
        return np.max(values)

    def _current_best_feasible(self):
        """Return best feasible sample so far."""
        if self.C is None or self.C.shape[0] == 0:
            return -np.inf
        
        feas_mask = np.all(self.C <= 0, axis=1)
        if not np.any(feas_mask):
            return -np.inf
        return np.max(self.y[feas_mask])

    def step(self, candidate_X: np.ndarray, iteration_num: int):
        """Perform one BO iteration using cKG acquisition."""
        acq_func= None
        if self.af=="cei":
            acq_func = ConstrainedExpectedImprovement(
                self.obj_model, self.cons_models, self._current_best_feasible()
            )
        elif self.af=="ckg":
            acq_func = ConstrainedKnowledgeGradient(
                self.obj_model, self.cons_models, candidate_X
            )
        
        print(f"--- Iteration {iteration_num}: Calculating {self.af} acquisition values... ---")
        acq_vals = np.array([acq_func(x) for x in candidate_X])
        
        if np.all(acq_vals <= 0):
            print(f"Warning: All {self.af} values <= 0. Picking a random candidate.")
            best_x_idx = np.random.randint(len(candidate_X))
        else:
            best_x_idx = np.argmax(acq_vals)
        
        best_x = candidate_X[best_x_idx]

        if self.plot_steps:
            self._plot_acquisition_and_gps(
                title=f"Iteration {iteration_num} - Next Sample: {best_x.round(3)}",
                acq_candidate_X=candidate_X,
                acq_vals=acq_vals,
                next_x_to_sample=best_x,
                filename=f"iter{iteration_num}.jpg"
            )
            
        y_new = self.obj_fn(best_x)
        c_new = np.array([fn(best_x) for fn in self.cons_fns])

        self.X = np.vstack([self.X, best_x.reshape(1, -1)])
        self.y = np.append(self.y, y_new)
        self.C = np.vstack([self.C, c_new.reshape(1, -1)])
        
        self.obj_model.fit(self.X, self.y)
        for k, gp in enumerate(self.cons_models):
            gp.fit(self.X, self.C[:, k])

        self.best_vals.append(self._current_best_feasible())
        self.iterations.append(iteration_num)

        return best_x, y_new, c_new

    def optimise(self, n_iter: int = 10, n_candidates: int = 100):
        """Run multiple BO iterations."""
        for i in range(1, n_iter + 1):
            candidate_X = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                            size=(n_candidates, self.bounds.shape[0]))
            self.step(candidate_X, i)
        
        self._plot_results()
        self._save_progress_json()
        
        return self.X, self.y, self.C

    def _save_progress_json(self):
        """Saves the optimization progress to a JSON file."""
        progress_data = {
            "iterations": self.iterations,
            "best_feasible_values": self.best_vals,
            "theoretical_max": self.theoretical_max,
            "samples": self.X.tolist(),
            "objective_values": self.y.tolist(),
            "constraint_values": self.C.tolist()
        }
        json_path = os.path.join(self.run_dir, "progress.json")
        with open(json_path, 'w') as f:
            json.dump(progress_data, f, indent=4)
        print(f"Saved progress data to: {json_path}")

    def _plot_results(self):
        """Plot and save final best feasible value and its gap to the theoretical maximum."""
        plt.figure(figsize=(7, 5))
        plt.grid(True, alpha=0.3)
        plt.title(f"Bayesian Optimisation Progress ({self.af})")
        plt.xlabel("Iteration")
        plt.ylabel("Objective value")
        plt.plot(self.iterations, self.best_vals, "o-", label="Current best (feasible)")
        if self.theoretical_max is not None:
            plt.axhline(self.theoretical_max, color="gray", linestyle=":", label=f"Theoretical max ({self.theoretical_max:.3f})")
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(self.run_dir, "progress.jpg")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved progress plot to: {save_path}")

    def _plot_acquisition_and_gps(self, title: str, 
                                  acq_candidate_X: np.ndarray,
                                  acq_vals: np.ndarray,
                                  next_x_to_sample: np.ndarray,
                                  filename: str):
        """Generates and saves plots for the current state of the BO."""
        if self.bounds.shape[0] != 2:
            print("Plotting is only supported for 2D problems.")
            return

        x1_min, x1_max = self.bounds[0]
        x2_min, x2_max = self.bounds[1]
        x1 = np.linspace(x1_min, x1_max, self.grid_density)
        x2 = np.linspace(x2_min, x2_max, self.grid_density)
        X_grid = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)

        Z_obj_true = np.array([self.obj_fn(x) for x in X_grid]).reshape(self.grid_density, self.grid_density)
        Z_c1_true = np.array([self.cons_fns[0](x) for x in X_grid]).reshape(self.grid_density, self.grid_density) if self.cons_fns else None
        mu_obj, _ = self.obj_model.predict(X_grid)
        Z_obj_gp_mean = mu_obj.reshape(self.grid_density, self.grid_density)
        pf_gp = self._get_gp_prob_feasible_grid(X_grid)
        Z_pf_gp = pf_gp.reshape(self.grid_density, self.grid_density)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        fig.suptitle(title, fontsize=16)

        # --- Plot 1: True Objective & Feasible Region ---
        ax = axes[0]
        c_true = ax.contourf(x1, x2, Z_obj_true.T, levels=20, cmap='viridis', alpha=0.8)
        fig.colorbar(c_true, ax=ax, label="True Objective Value")
        if Z_c1_true is not None:
            ax.contour(x1, x2, Z_c1_true.T, levels=[0], colors='white', linestyles='--', linewidths=2)
            ax.contourf(x1, x2, Z_c1_true.T, levels=[0, np.inf], colors='none', hatches=['//'], alpha=0.3)
        ax.set_title("True Objective & Feasible Region")
        ax.set_ylabel("x2")

        # --- Plot 2: GP Predictions ---
        ax = axes[1]
        c_gp_obj = ax.contourf(x1, x2, Z_obj_gp_mean.T, levels=20, cmap='viridis', alpha=0.8)
        fig.colorbar(c_gp_obj, ax=ax, label="GP Predicted Mean Objective")
        pf_contour = ax.contour(x1, x2, Z_pf_gp.T, levels=[0.5, 0.9], colors='cyan', linestyles='-', linewidths=1.5)
        ax.clabel(pf_contour, inline=True, fontsize=10)
        ax.set_title(f"GP Mean Objective & Prob. Feasible (PF)")

        # --- Plot 3: Acquisition ---
        ax = axes[2]
        ax.set_title(f"{self.af} Acquisition (on Candidates)")
        if acq_vals is not None and acq_candidate_X is not None:
            scatter = ax.scatter(acq_candidate_X[:, 0], acq_candidate_X[:, 1],
                                 c=acq_vals, cmap='Reds', marker='o', s=50, edgecolors='gray')
            fig.colorbar(scatter, ax=ax, label=f"{self.af} Acquisition Value")

        # --- Overlay sampled points on all plots ---
        for ax in axes:
            ax.set_xlabel("x1")
            if self.X is not None and len(self.X) > 0:
                feas_mask = np.all(self.C <= 0, axis=1)
                ax.scatter(self.X[feas_mask, 0], self.X[feas_mask, 1],
                           c='lime', marker='o', s=80, edgecolors='black', label="Feasible Samples", zorder=5)
                ax.scatter(self.X[~feas_mask, 0], self.X[~feas_mask, 1],
                           c='red', marker='X', s=80, edgecolors='black', label="Infeasible Samples", zorder=5)
            
            if next_x_to_sample is not None:
                ax.scatter(next_x_to_sample[0], next_x_to_sample[1], 
                           c='cyan', marker='*', s=300, edgecolors='black', 
                           label=f"Next Sample ({self.af})", zorder=10)
        
        handles, labels = axes[0].get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(self.run_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved iteration plot to: {save_path}")

    def _get_gp_prob_feasible_grid(self, X_grid: np.ndarray) -> np.ndarray:
        """Helper to get current GP-estimated probability of feasibility for a grid."""
        if not self.cons_models:
            return np.ones(len(X_grid))
        pf_total = np.ones(len(X_grid))
        for gp in self.cons_models:
            mu_c, var_c = gp.predict(X_grid)
            var_c = np.clip(var_c, 1e-12, None)
            pf_total *= norm.cdf(-mu_c / np.sqrt(var_c))
        return pf_total
# ===========================================================
# 5. Example usage
# ===========================================================
def f_obj_G(x):
    """
    Multimodal G-Function Objective (Maximization).
    The global maximum is located at a hard-to-find point close to the boundary.
    The true unconstrained maximum is around 1.05.
    """
    x0, x1 = x[0], x[1]
    
    # We negate the standard objective for minimization and add an offset for clarity.
    # The true objective function for the G-problem is complex; this is a common G-like test function.
    
    # This specific form introduces high frequency oscillations for complexity:
    value = 10 * np.sin(np.pi * x0) * np.cos(2 * np.pi * x1) \
            + np.exp(-x0 / 2) * np.cos(np.pi * x1) + 0.5
            
    return 1.05 - value

def c1_G(x):
    """
    Non-linear, tight constraint c1(x) <= 0.
    Feasible region is defined by 0.5 - [1/2 * (sin(3*pi*x0) + sin(3*pi*x1))] <= 0
    """
    x0, x1 = x[0], x[1]
    
    # Constraint expression: Must be less than or equal to zero for feasibility
    return 0.5 - 0.5 * (np.sin(3 * np.pi * x0) + np.sin(3 * np.pi * x1))

if __name__ == "__main__":
    # Objective and constraint test functions
    def f_obj(x):
        # Peak at (1, 0.5) with value 1.0
        return -(x[0]-1.1)**2 - (x[1]-1.1)**2 + 1

    def c1(x):
        # Linear constraint: x0 + x1 - 1.2 <= 0 is feasible
        return x[0]**3 + x[1]**2 - 2

    bounds = np.array([[0, 2], [0, 2]])

    # Initialize with plot_steps=True to see the visualizations per step
    # You can set grid_density lower for faster plotting, higher for smoother plots
    bo = BayesianOptimizer(f_obj_G, [c1_G], bounds, plot_steps=True, grid_density=40) 
    
    # Increase initial points to better cover the space
    bo.initialise(n_init=5, af= "cei") 
    X, y, C = bo.optimise(n_iter=15, n_candidates=100) # Run fewer iterations for clarity with plots

    print("Final samples:")
    for i, (xi, yi, ci) in enumerate(zip(X, y, C)):
        print(f"Sample {i+1}: x={xi}, y={yi:.3f}, c={ci}")