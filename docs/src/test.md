
# Testing the Solver
Dynamic portfolio optimization using simulation and cross-sectional regressions is mathematically intensive. Errors can easily hide in floating-point arithmetic, matrix inversions, or the backward induction loops.

To ensure `BrandtSolver.jl` produces economically sound and mathematically precise results,this page explains the economic theory and choices behind our core integration tests.

## 1. The Merton Benchmark
**The Theory:** In 1969, Robert Merton proved that if an investor with Constant Relative Risk Aversion (CRRA) $\gamma$ faces a constant investment opportunity set (a constant risk-free rate $r$, constant stock drift $\mu$, and constant volatility $\sigma$), the optimal portfolio weight in the risky asset is completely myopic and constant over time:
```math
\omega^* = \frac{\mu - r}{\gamma \sigma^2}
```
### The Test (`validity_test.jl`):
To test this, we simulate a market environment designed to mimic Merton's continuous-time assumptions within our discrete-time framework. We utilize a Vasicek interest rate model with an extremely tiny volatility ($0.001$) so it acts as a constant rate.
We then couple this with a Geometric Brownian Motion stock.
We run the full simulation-based Brandt solver on this data. If the regression mechanics, Taylor expansions, and polynomial optimizers are correct, the complex simulation algorithm must collapse back to the simple analytical Merton fraction.

This proves that the numerical solver is fundamentally unbiased in a baseline environment and successfully recovers known closed-form solutions.

## 2. The Economic Perturbation Test
By definition, the optimal portfolio weight $\omega^*$ returned by the First Order Conditions must maximize the expected utility of the investor.
Therefore, any manual deviation from this optimal weight—no matter how small—must strictly result in a lower expected utility.

### The Test (validity_test.jl):
Once the solver calculates the optimal policies, we perform a "Perturbation Check".
We define a small constant perturbation $\epsilon = 0.05$ and calculate the Expected Utility for three scenarios:
- $J(\omega^*):$ Using the solver's optimal weights.
- $J(\omega^* + \epsilon):$ Forcing the agent to slightly over-invest in the risky assets.
- $J(\omega^* - \epsilon):$ Forcing the agent to slightly under-invest in the risky assets.

The test dynamically verifies that $J(\omega^*) > J(\omega^* + \epsilon)$ and $J(\omega^*) > J(\omega^* - \epsilon)$.
This guarantees that the non-linear root-finding algorithm has actually identified a true utility maximum, rather than converging to a local minimum or an inflection point.

## 3. Value Function vs. Terminal Utility
The Bellman equation dictates that at the terminal time $M + 1$, the Value Function $V_M(W_{M + 1})$ is exactly equal to the utility function $u(W_{M + 1})$.
Therefore, at time $M$, the conditionally expected utility of optimally trading one last time should smoothly converge to the shape of the theoretical utility function, sitting slightly above it to reflect the value added by that final trade.

### The Test (visual_test.jl):
Using our `plot_value_vs_utility` tool, we forward-simulate the portfolio from $M$ to $M + 1$ applying the optimal policy.
We plot the resulting conditionally expected utility against the theoretical $U(W)$ curve.
This acts as a boundary condition check.
If the regression was unstable or the Taylor expansion was poorly specified, the predicted expected utility would wildly diverge from the true $U(W)$ curve.

## 4. Robustness to Singularities
When multiple assets are highly correlated, the covariance matrix of their returns becomes singular (its determinant approaches zero). In the Taylor-expanded First Order Conditions, generating the optimal initial guess requires inverting a matrix $B_n$ which approximates this second-moment structure.
If two assets are virtually identical, standard matrix inversion will fail.

### The Implementation (`optimizer.jl` & `optimizer_test.jl`):
To ensure the solver is robust in pathological market environments (like identical assets), `BrandtSolver.jl` dynamically calculates the scale of the matrix trace and injects a heavily scaled Tikhonov regularization term (jitter $\lambda I$) into the diagonal of the $B_n$ matrix:

```julia
mean_diag = mean(abs.(diag(B_mat)))
scale = mean_diag > 1e-20 ? mean_diag : 1.0
λ = 1.0e-6 * scale
jit = λ * I
```
This allows the solver to smoothly handle environments with extreme correlations or redundant assets without throwing `SingularException` errors, safely distributing weights across the correlated assets.