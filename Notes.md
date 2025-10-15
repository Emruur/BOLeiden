# Knowledge Gradient for Constrained Problems

## Problem with acquisition functions

- Regular aquisition functions aim to find the maximum in a single step

- However, BO is a multi step iterative process. Sampling at a point provides some information to later time steps. 

- Regular acquisition functions do not fully capture the value of the information gained from sampling a solution.

- Knowledge gradient simulates a one step lookahead from our sampling point $x^{n+1}$ and compares the highest point of our current acquisition function with the new highest value *given the decision to take the next sample at $x^{n+1}$

- If I sample next at point $x^{n+1=x}$
what’s the expected increase in my best predicted value after updating the GP

## Formal Definition

$$
\mathrm{KG}(x) = 
\mathbb{E}\!\left[
    \max_{x'' \in \mathcal{X}} 
        \{ \mu_y^{n+1}(x'') \}
    - 
    \max_{x' \in \mathcal{X}}
        \{ \mu_y^{n}(x') \}
    \;\middle|\;
    x^{n+1} = x
\right].
$$

- Lets hypothesize that we choose x as our new sampling point
$x^{n+1} = x$

- According to our GP model we could have diferent realisations of $y^{n+1}$ thus the outer expectation

- Information gain is the difference bw old maximum: $\max_{x' \in \mathcal{X}}
        \{ \mu_y^{n}(x') \}$ and the new maximum $\max_{x' \in \mathcal{X}}
        \{ \mu_y^{n}(x') \}$ after we update our GP with the new $y^{n+1}$ realisation

- How to calculate this?? max operators and the Expectation are both on a continuous domain

## How to calculate efficiently

### Monte Carlo Sampling WuandFrazier2017

- Draw a monte carlo sample for $y^{n+1}$, update your gp and identify its new maximum

### Discretization and Linear-Envelope Scott et al 2011

#### Reparametirasition trick

$$
\mu_y^{n+1}(x)
= \mu_y^{n}(x) \;+\; \tilde{\sigma}_y^{\,n}(x, x^{n+1}) \, Z_y,
\qquad Z_y \sim \mathcal{N}(0,1),
$$

- $ \tilde{\sigma}_y^n(x, x^{n+1}) $: correlation between x and $ x^{n+1}$. How strongly the new observation at $x^{n+1}$ affects point $x$.  


- The future mean at $x$ is the current mean at $x$,
**plus** a random correction that depends on how correlated $x$ is with the new point $x^{n+1}$.

- Notice the new posterior mean at x  $\mu_y^{n+1}(x)$ is a linear function of $Z_y$

### Discretizing the domain

- The domain $ \mathcal{X} $ is discretized into a finite set $X_j$ $ \{x_1, x_2, \dots, x_J\} $.  

- For each discrete point $x_j$, an **updated** posterior mean formulation is obtained using the reparameterization trick:
$$
  \mu_y^{n+1}(x_j) = \mu_y^n(x_j) + \tilde{\sigma}_y^n(x_j, x^{n+1}) Z_y,
  \quad Z_y \sim \mathcal{N}(0,1).
$$
  Each $x_j$ therefore corresponds to a **linear function of** $Z_y$.

- By taking the **maximum** across these linear functions, we obtain a **piecewise-linear function** that represents  
$$
  \max_{x' \in \mathcal{X}_d} \{ \mu_y^{n+1}(x') \}
$$
  as a function of $Z_y$.

- Finally, we integrate this piecewise-linear function with respect to the standard normal density of $Z_y$ to compute  
$$
  \mathbb{E}_{Z_y}\!\left[ \max_{x' \in \mathcal{X}_d} \{ \mu_y^{n+1}(x') \} \right].
$$

### High Value Estimates Pearce et al. [2020]

- Build on the idea of Scott et al 2011 
- They propose a different way of discretizing the domain X
- Rather than using the fixed $X_j$  obtain high value points from the predictive posterior mean GP that would serve as
a discretisation.

### Obtain high-value points

- Select a few quantiles of the standard normal variable $ Z_y $, for example $ Z_y \in \{-1, 0, 1\} $.  
  These correspond to possible realizations of the future observation $ y^{n+1} $.

- For each sampled value of $ Z_y $, update the GP posterior using  
$$
  \mu_y^{n+1}(x) = \mu_y^{n}(x) + \tilde{\sigma}_y^{n}(x, x^{n+1}) Z_y,
$$
  and obtain the point $ x_j^* $ that maximizes this updated mean:
$$
  x_j^* = \arg\max_{x \in \mathcal{X}} \mu_y^{n+1}(x).
$$

- Collect all such maximizing points into a discrete set
$$
  \mathcal{X}_d = \{ x_1^*, x_2^*, \dots, x_J^* \}.
$$


Continue with Scott et al. (2011) using the scenario-driven discrete set $ \mathcal{X}_d $:

- For each $ x_j^* \in \mathcal{X}_d $, build a linear function in $ Z_y $:
$$
  \mu_y^{n+1}(x_j^*) = \mu_y^n(x_j^*) + \tilde{\sigma}_y^n(x_j^*, x^{n+1}) Z_y.
$$

- Take the **maximum** across these linear functions to form a **piecewise-linear** function of \( Z_y \):
$$
  L(Z_y) = \max_{x_j^* \in \mathcal{X}_d} \mu_y^{n+1}(x_j^*).
$$

- Integrate $ L(Z_y) $ with respect to the standard normal density to obtain  
$$
  \mathbb{E}_{Z_y}[L(Z_y)].
$$

Unlike the original fixed discretization, this **scenario-driven discrete set** does not grow with the number of dimensions.

## Constrained BO Definition



$$
x^* = \arg\max_{x \in \mathcal{X}} f(x)
\quad \text{s.t.} \quad
c_k(x) \le 0, \quad k = 1, \dots, K.
\tag{1}
$$

---

### Gaussian Process Modeling

Both the objective $ f(x) $ and the constraints $ c_k(x) $ are modeled as independent Gaussian Processes (GPs).   At iteration $ n $, their posteriors are described by their respective means and variances:
$$
\mu_y^n(x), \; \sigma_y^n(x) \quad \text{for the objective, and} \quad
\mu_{c_k}^n(x), \; \sigma_{c_k}^n(x) \quad \text{for each constraint.}
$$

---

### Probability of Feasibility (PF)

The **probability of feasibility** for a candidate design $ x $ is defined as the probability that all constraints are satisfied under the current GP models:

$$
\text{PF}^n(x)
= \prod_{k=1}^{K} P(c_k(x) \le 0)
= \prod_{k=1}^{K} \Phi\!\left(-\frac{\mu_{c_k}^n(x)}{\sigma_{c_k}^n(x)}\right),
$$


---

### Recommended Design - What to optimize?

The **penalized posterior mean** — the GP mean of the objective scaled by the probability of feasibility:

$$
x_r^n
= \arg\max_{x \in \mathcal{X}}
\; \mu_y^n(x) \, \text{PF}^n(x).
$$

This formulation balances exploration of promising regions with the likelihood of satisfying constraints.

## Formal Constrained KG definition

### Constrained Knowledge Gradient (cKG)

The constrained Knowledge Gradient (cKG) acquisition function quantifies the **expected gain in feasible objective performance** after sampling at a candidate point $ x^{n+1} = x $:

$$
\text{cKG}(x)
=
\mathbb{E}\!\left[
\max_{x' \in \mathcal{X}}
\big\{ \mu_y^{n+1}(x') \, \text{PF}^{n+1}(x') \big\}
-
\mu_y^n(x_r^n) \, \text{PF}^{n+1}(x_r^n)
\;\middle|\;
x^{n+1} = x
\right].
\tag{9}
$$

## Reparametrisation 

Recall the reparameterisation trick

- **Objective:**
$$
  y(x) = \mu_y^n(x) + \tilde{\sigma}_y(x, x^{n+1}) Z_y, \quad Z_y \sim \mathcal{N}(0, 1)
$$

- **Constraints:**
$$
  c(x) = \mu_c^n(x) + \tilde{\sigma}_c(x, x^{n+1}) Z_c, \quad Z_c \sim \mathcal{N}(0, I)
$$

- **Probability of feasability**
The term $ \mathrm{PF}^{n+1}(\cdot; Z_c) $ stands for the **Probability of Feasibility** after the $(n+1)$-th evaluation.

- It represents the probability that a candidate $ x $ satisfies all constraints, given a random constraint outcome parameterised by $ Z_c $:
$$
  \mathrm{PF}^{n+1}(x; x^{n+1}, Z_c) = \Pr\big[c_j(x) \le 0 \; \forall j \,\big|\, Z_c, \mathcal{D}^{n+1}\big].
$$




$$
\mathrm{cKG}(x) = 
\mathbb{E}_{Z_c, Z_y} \left[
\underbrace{
\max_{x' \in \mathcal{X}} 
\left\{
\left[ 
\mu_y^n(x') + \tilde{\sigma}_y(x', x^{n+1}) Z_y 
\right]
\mathrm{PF}^{n+1}(x'; x^{n+1}, Z_c)
\right\}
}_{\text{New best penalized posterior mean } }
- 
\mu_y^n(x_r)
\mathrm{PF}^{n+1}(x_r^n; x^{n+1}, Z_c)
\;\middle|\;
x^{n+1} = x
\right].
$$

How to compute this?

- First find $ x_r^n $ using a continuous numerical optimiser.  

- Then, we generate a discretisation $ \mathcal{X}_d $ given a design $ x^{n+1} $.  
This is done using $ n_y $ values from $ Z_y $ and $ n_c $ values from $ Z_c $, where the inner optimisation problems in Equation (10) are solved by a continuous numerical optimiser for all $ n_z = n_c \times n_y $ values.  


-  Furthermore, conditioned on Zc, the expectation in Equation (10) can be seen as marginalising
thestandard KG over the constraint uncertainty


$$
\mathrm{cKG}(x) = 
\mathbb{E}_{Z_c} 
\left[
  \mathbb{E}_{Z_y} 
  \left[
    \max_{x' \in \mathcal{X}}
    \left\{
      \left[\mu_y^n(x') + \tilde{\sigma}_y(x', x^{n+1}) Z_y\right]
      \mathrm{PF}^{n+1}(x'; x^{n+1}, Z_c)
    \right\}
    -
    \mu_y^n(x_r)
    \mathrm{PF}^{n+1}(x_r^n; x^{n+1}, Z_c)
    \;\middle|\;
    x^{n+1} = x, Z_c
  \right]
\right].
$$

Here, $ \mu_y^n(x) $ and $ \tilde{\sigma}_y(x, x^{n+1}) $ are penalised by the deterministic feasibility function $ \mathrm{PF}^{n+1}(x; x^{n+1}, Z_c) $.

The **inner expectation** over $ Z_y $ can be solved in closed form over the discrete set $ \mathcal{X}_d $ using the **discrete KG algorithm** $ \mathrm{KG}_d $ (Scott et al., 2011).  
The **outer expectation** over $ Z_c $ is then computed by a **Monte Carlo approximation**, taking the average over $ n_c $ different $ \mathrm{KG}_d $ computations:
$$
\mathrm{cKG}(x) = \frac{1}{n_c} \sum_{m=1}^{n_c} \mathrm{KG}_d(x^{n+1} = x; Z_c^m).
$$


## CKG computation pseudocode  

**Input:** Sample $ x^{n+1} $, size of Monte Carlo discretisations $ n_c $ and $ n_y $

0. Initialise discretisation $ X_d^0 = \{\} $ and set $ n_z = n_c n_y $  
1. Compute $ x_r^n = \arg\max_{x \in \mathcal{X}} \mu_y^n(x) \mathrm{PF}^n(x) $  
2. **for** $ j \in [1, \ldots, n_z] $:  
   3. Generate $ Z_y^j, Z_1^j, \ldots, Z_K^j \sim \mathcal{N}(0, 1) $  
   4. Compute  
$$
      x_j^* = \max_{x \in \mathcal{X}_d}
      \left\{
        [\mu_y^n(x) + \tilde{\sigma}_y(x, x^{n+1}) Z_y^j]
        \mathrm{PF}^{n+1}(x; x^{n+1}, Z_c^j)
      \right\}
$$
   5. Update discretisation $ X_d^j = X_d^{j-1} \cup \{x_j^*\} $  
6. **for** $ m \in [1, \ldots, n_c] $:  
   7. Compute $ \mathrm{KG}_d(x^{n+1} = x; Z_c^m) $ using $ X_d $  
8. Compute Monte Carlo estimation  
$$
   \frac{1}{n_c} \sum_{m=1}^{n_c} \mathrm{KG}_d(x^{n+1} = x; Z_c^m)
$$
9. **Return:** $ \mathrm{cKG}(x^{n+1}) $










# Notes

