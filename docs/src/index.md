# BrandtSolver.jl

A high-performance Julia package for solving discrete-time dynamic portfolio choice problems
using the simulation-based method originally proposed by [Brandt_etal_2005](@cite).

In many realistic financial models (e.g., stochastic interest rates, predictability in
asset returns, non-tradable income), analytical solutions to portfolio choice problems are hard to come by.
`BrandtSolver.jl` overcomes this by combining already existing simulations of asset paths with cross-sectional regressions
to dynamically approximate the optimal portfolio policy backwards through time.

More specifically, in this package we consider portfolio choice problems at timesteps
$n = 1, 2, \ldots, M$, where $M + 1$ is some terminal timestep.
This portfolio choice problem at timestep $n$ is defined by an investor who maximizes the expected utility
of their wealth at the terminal timestep $M + 1$ by trading $N$ risky assets and a risk-free asset (cash).
Formally the investor's problem at timestep $n$ is
```math
    V_n(W_n, Z_n)
    = \max_{\{\omega_s\}_{m = n}^{M}} \mathbb{E}_n[u(W_{M + 1})]
```
subject to the sequence of budget constraints
```math
    W_{m + 1} = W_m (\omega_m^\top R^e_{m + 1} + R_{m + 1})
```
for all $m \geq n$.
Here $R^e_{m + 1}$ can be interpreted as the excess return of the risky assets over the risk-free
asset, and $R_{m + 1}$ is the gross return of other processes that _may_ depend on wealth
$W_m$.
Furthermore, $\{\omega_s\}_{m=n}^{M}$ is the sequence of portfolio weights chosen at times
$m = n, \ldots, M$ and $u$ is the investor's utility function.
The process $Z_n$ is a vector of state variables that are relevant for the investor's decision making.
The goal of this package is to find $\{\omega_m\}_{m=1}^{M}$.

### Extension: Wealth-Dependent Returns
A key feature of this implementation is that the gross return $R_{m+1}$ is not restricted to be exogenous.
We allow $R_{m+1}$ to depend on the current level of wealth $W_m$ through the following structure:
```math
    R_{m+1} = X_m + \frac{Y_m}{W_m}
```
where $X_m$ and $Y_m$ are functions of state variables contained in $Z_m$.
This formulation is particularly powerful as it allows the algorithm to incorporate
**non-tradeable income or fixed costs**.
For instance, if $Y_m$ represents labor income, the budget constraint correctly captures that
income is added to wealth regardless of the portfolio choice
$\omega_m$.

#### Example
Consider an investor who receives a stochastic labor income $O_n$ at each timestep and saves a
proportion $p \in [0,1]$ of that income.
Let $R^f_n$ be the gross risk-free rate.
The wealth at time $n+1$ is:
```math
    W_{n+1} = W_n (\omega_n^\top R^e_{n+1} + R^f_n) + p O_n
```
By setting $X_n = R^f_n$ and $Y_n = p O_n$,
this matches our budget constraint $W_{n+1} = W_n (\omega_n^\top R^e_{n+1} + R_{n+1})$.


## Features
- **Simulation-Based**: Solves complex, multi-period portfolio optimization problems using
    generated market scenarios.
- **State-Dependent Policies**: Computes optimal portfolio weights that dynamically depend on
    exogenous state variables (e.g., interest rates) and the agent's current wealth.
- **Arbitrary Utility Functions**: Leverages Automatic Differentiation (`ForwardDiff.jl`)
    to automatically compute exact higher-order Taylor expansions for any custom utility function.

## Installation
You can install the package via the Julia REPL:
```julia
using Pkg
Pkg.add(url="https://github.com/sliemelela/BrandtSolver.jl")
```

## Quick start

```julia
using BrandtSolver
using FinancialMarketSimulation # (Optional, for generating paths)

# 1. Define Solver Configuration
params = SolverParams(
    W_grid = [50.0, 100.0, 150.0], # Wealth grid to evaluate
    poly_order = 2,                # Polynomial expansion of state variables
    max_taylor_order = 4,          # Value function Taylor expansion order
    trimming_α = 0.01              # Trim extreme 1% of paths during regression
)

# 2. Define Utility
crra(W) = (W^(1.0 - 5.0)) / (1.0 - 5.0)
utility = create_utility_from_ad(crra)

# 3. Extract Matrices from your Simulation
# Re_all: Risky Asset Excess Returns
# Z_all: Predictor State Variables
# X_all, Y_all: Risk-Free Return and Income Components
Re_all = package_excess_returns(world, ["Re_Stock"])
Z_all  = get_state_variables(world, ["r"])
X_all, Y_all = create_risk_free_return_components(world, 0.0, nothing)

# 4. Solve the Dynamic Program!
policies = solve_portfolio_problem(Re_all, Z_all, X_all, Y_all, params, utility)
```

For an explanation of the above code, see [Tutorial](tutorial.md).
## References
```@bibliography
```