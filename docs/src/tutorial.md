# Tutorial: Solving a Dynamic Portfolio Choice Problem

In this tutorial, we will use `BrandtSolver.jl` alongside `FinancialMarketSimulation.jl` to solve a classic dynamic portfolio choice problem. We expect the reader to have read the introduction in [Home](index.md) to understand the notation and context.

We will model an investor with Constant Relative Risk Aversion (CRRA) utility who must allocate their wealth between a risk-free asset with a stochastic interest rate and a risky stock.
Because the interest rate is stochastic and correlated with the stock, the investor will exhibit *intertemporal hedging demand*—their optimal allocation will depend on the current interest rate!

## Step 1: Simulating the Market
First, we need to generate the "paths" of the world. We use `FinancialMarketSimulation.jl` to build a market with a mean-reverting Vasicek interest rate and a stock whose drift depends on that rate. You can of course feel free to generate in some other fashion.

```julia
using BrandtSolver
using FinancialMarketSimulation
using StaticArrays

# Simulation parameters
sims = 1000
T = 1.0
M = 12

# Correlation of shocks
ρ_rS = -0.5
ρ = [ 1.0 ρ_rS;
     ρ_rS  1.0]

# Market prices of risk (defined via factor loadings)
ϕ_r, ϕ_S = 0.075, -0.2
λ_r = -ϕ_r - ϕ_S * ρ_rS
λ_S = -ϕ_S - ϕ_r * ρ_rS

# Define the interest rate (Vasicek)
κ_r, θ_r, σ_r, r_0 = 0.5, 0.05, 0.03, 0.05
idx_r_shock = 1
r_proc = VasicekProcess(:r, κ_r, θ_r, σ_r, r_0, idx_r_shock)

# Define the Stock
σ_S = 0.20
stock_drift(t, S, r_val) = S * (r_val + λ_S * σ_S)
stock_diff(t, S, r_val)  = S * σ_S
stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, 100.0, [2], [:r])

# Obtain the excess return process
re_proc = ExcessReturnProcess(:Re_Stock, :S, :r)
config = MarketConfig(sims = sims, T = T, M = M, processes = [r_proc, stock_proc, re_proc])
world = build_world(config)

# Define the Excess Return Process
re_proc = ExcessReturnProcess(:Re_Stock, :S, :r)

# Generate the market
config = MarketConfig(
    sims = 2000,
    T = T,     # 1 years
    M = M,     # Monthly rebalancing
    processes = [r_proc, stock_proc, re_proc],
    correlations = ρ
)
world = build_world(config)
```

## Step 2: Configuring the Solver
Next, we define the parameters for our dynamic programming solver. We need to specify the wealth grid over which to evaluate the policy, the order of our Taylor expansions, and our utility function.

```julia
# Setup Solver Parameters
params = SolverParams(
    W_grid = [50.0, 100.0, 150.0], # Wealth nodes to interpolate between
    poly_order = 2,                # Use up to Z^2 in regressions
    max_taylor_order = 4,          # 4th-order Taylor expansion of the value function
    trimming_α = 0.01              # Discard extreme 1% of paths to stabilize regressions
)

# Define Utility using Automatic Differentiation
γ = 5.0
crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
utility = create_utility_from_ad(crra)
```

## Step 3: Formatting Data for the Solver
Recall that the budget constraint that `BrandtSolver.jl` uses is
```math
    W_{m + 1} = W_m (\omega_m^\top R^e_{m + 1} + R_{m + 1})
```
with
```math
    R_{m+1} = X_m + \frac{Y_m}{W_m}.
```
An example of what $X_m$ and $Y_m$ represents can be found in the introduction in [Home](index.md).
The reason why the budget constraint is formulated in this form is highlighted in [Tutorial](tutorial.md).
Now note that $X_m$, $Y_m$ are real-valued, while $R^e_{m + 1}$ is vector valued with $N$ entries, with $N$ being the amount of risky assets that are traded.

`BrandtSolver.jl` expects the simulation data in specific matrix formats.
Specifically, we expect $\{X_m\}_{m = 1}^{M + 1}$ and $\{Y_m\}_{m = 1}^{M + 1}$ to be represented as $S \times (M + 1)$ matrices, where $S$ are the amount of simulations and $M + 1$
are the amount of timesteps, i.e. we expect $\{X_m\}_{m = 1}^{M + 1}$ to be represented as
```math
    \{X_m\}_{m = 1}^{M + 1} \triangleq
    \begin{pmatrix}
    X_{1}^{s=1} & \cdots & X_{M + 1}^{s=1} \\
    X_{1}^{s=2} & \cdots & X_{M + 1}^{s=2} \\
    \vdots      & \ddots & \vdots          \\
    X_{1}^{s=S} & \cdots & X_{M + 1}^{s=S}
    \end{pmatrix},
```
and $\{Y_m\}_{m = 1}^{M + 1}$ in a similar manner.
The process $\{R^{e}_{m + 1}\}_{m = 1}^{M + 1}$ is also expected to be an
$S \times (M + 1)$ matrix, but with entries being vectors with $N$ entries, i.e.
```math
    \{R^{e}_{m + 1}\}_{m = 1}^{M + 1}\triangleq
    \begin{pmatrix}
    \begin{pmatrix} \left(R^{e, 1}_{1}\right)^{s=1} \\ \vdots \\ \left(R^{e, N}_{1}\right)^{s=1} \end{pmatrix} & \cdots & \begin{pmatrix}\left(R^{e, 1}_{M + 1}\right)^{s=1} \\ \vdots\\ \left(R^{e, N}_{M + 1}\right)^{s=1} \end{pmatrix} \\
        \begin{pmatrix} \left(R^{e, 1}_{1}\right)^{s=2} \\ \vdots \\ \left(R^{e, N}_{1}\right)^{s=2} \end{pmatrix} & \cdots & \begin{pmatrix}\left(R^{e, 1}_{M + 1}\right)^{s=2} \\ \vdots \\ \left(R^{e, N}_{M + 1}\right)^{s=2} \end{pmatrix} \\
    \vdots      & \ddots & \vdots          \\
     \begin{pmatrix} \left(R^{e, 1}_{1}\right)^{s=S} \\ \vdots \\ \left(R^{e, N}_{1}\right)^{s=S} \end{pmatrix} & \cdots & \begin{pmatrix}\left(R^{e, 1}_{M + 1}\right)^{s=S} \\ \vdots \\ \left(R^{e, N}_{M + 1}\right)^{s=S} \end{pmatrix} \\
    \end{pmatrix},
```
The processes $\{X_m\}_{m = 1}^{M + 1}, \{Y_m\}_{m = 1}^{M + 1}$, $\{R^{e}_{m + 1}\}_{m = 1}^{M + 1}$ will be referred to as `X_all_paths`, `Y_all_paths` and `Re_all_paths` respectively in our Julia code. We extract these directly from our simulated world from the previous step.

The solver function is given by `solve_portfolio_problem` which accepts the following arguments in the following order:
- `Re_all_paths::AbstractMatrix{<:SVector}`: A matrix (`sims × M`) of `SVector`s representing excess returns.
- `X_all_paths::AbstractMatrix{Float64}`: A matrix (`sims × M`) of gross risk-free returns.
- `Y_all_paths::AbstractMatrix{Float64}`: A matrix (`sims × M`) of non-tradable income yields.
- ​`solver_params::SolverParams`: Configuration parameters (e.g., W_grid, polynomial orders).
- `utility::UtilityFunctions`: The configured utility and derivatives functions.

Let us extract them with some interface functions that can be found in `test/test_helpers.jl`.
For now you may simply assume that we will get the desired formats of the matrices.

```julia
# Define the names of the assets and state variables we care about
asset_names = ["Re_Stock"]
state_names = ["r"]

# Extract and package the matrices
Re_all = package_excess_returns(world, asset_names)
Z_all  = get_state_variables(world, state_names)

# Generate Risk-Free Returns (X) and non-tradable income (Y).
# We assume no exogenous income here (p_income = 0.0).
X_all, Y_all = create_risk_free_return_components(world, 0.0, nothing)
```

## Step 4: Solving the Problem
With the market simulated and the parameters configured, we can trigger the backward-recursive solver.

```julia
# Run the dynamic programming solver
policies = solve_portfolio_problem(Re_all, Z_all, X_all, Y_all, params, utility)
```
The policies object is a nested vector containing the interpolated optimal portfolio weights. You can query the optimal weight at a specific time step t for a specific path i at a specific wealth W like this: `policies[t][i](W)`.

## Step 5: Visualizing the Results
`BrandtSolver.jl` comes with a comprehensive suite of plotting tools powered by `Makie.jl`. Because dynamic portfolio optimization produces complex, state-dependent, and time-varying rules, these visualizations are essential for understanding the investor's behavior and validating the economic logic of the results.

First, ensure you have a Makie backend loaded (we use `CairoMakie` for static 2D and 3D plots) and select a time step to analyze:

```julia
using CairoMakie

# Let's inspect the policies at the penultimate time step
t_check = M - 1
```

### 1. Policy as a Function of Wealth
The most fundamental question is: How much should the investor allocate to the risky asset given their current wealth?
```julia
# Plot 1: Policy as a function of Wealth
fig1 = plot_policy_rules(policies, params, t_check, asset_names; samples=50)
```

**What it shows:** This function plots the optimal portfolio weight ($\omega$) against the wealth grid ($W$).
- **Cross-Sectional Dispersion:** It randomly selects a subset of individual simulation paths (controlled by the `samples` keyword argument) and plots them as semi-transparent blue lines. The vertical spread between these lines at any given wealth level highlights how the optimal weight changes due to the different *state variables* across the simulated paths.
- **Mean Policy:** A thick red line overlays the plot, representing the average optimal policy across all simulations.


### 2. Policy as a Function of the State Variable (Intertemporal Hedging)
To isolate *why* the individual paths in the previous plot were dispersed, we can hold wealth strictly constant and observe the policy's sensitivity to the state variable.

```julia
# Plot 2: Policy as a function of the State Variable (Interest Rate)
state_vals = world.paths.r[:, t_check]
fig2 = plot_state_dependence(policies, params, t_check, state_vals, "r", asset_names; fix_W=100.0)
```

**What it shows**: This plot fixes wealth at fix_W=100.0 and plots the optimal weight against the realized state variable (the interest rate r).

#### Economic Intuition
This directly visualizes the intertemporal hedging demand.
Because the interest rate is correlated with the stock, the investment opportunity set changes as the rate fluctuates.
This plot shows exactly how the investor adjusts their portfolio to hedge against these changing opportunities.

### 3. Forward Simulation of Realized Weights
Rather than looking at a static snapshot, we can watch how the investor behaves over their entire lifecycle.

```julia
# Plot 3: Forward simulation of the realized weights over time
times = range(0, world.config.T, length=world.config.M+1)[1:end-1]
fig3 = plot_realized_weights(Re_all, X_all, Y_all, times, policies, asset_names; W_init=100.0)
```
**What it shows**: This function simulates the actual wealth trajectories forward in time from $t=0$ to $T$, applying the computed optimal policies at each step.

- **Lifecycle Dynamics**: It plots the realized weight paths over time (semi-transparent blue lines), overlaying the mean path (black line) and a 90% confidence interval band.
- **Robustness**: If bad market shocks drive an investor's wealth to zero, the function dynamically filters out these bankrupt paths (assigning them NaN) to prevent the statistics from exploding.

### 4. Economic Validity: Value Function vs. Utility
To mathematically verify that our dynamic solver adds value, we can compare the expected outcomes of our strategy against the baseline utility.

```julia
# Plot 4: Economic Validity Check
fig4 = plot_value_vs_utility(Re_all, X_all, Y_all, params, policies, utility; t_check=t_check)
```
What it shows: This acts as an economic validity check.
It plots the forward-simulated expected utility (the Value function, $J(W)$) against the theoretical terminal utility function $U(W)$.
- **Value of Optimization**: At timestep $M$, $J(W)$ should naturally sit above or equal to $U(W)$. The gap between the dashed theoretical curve and the predicted curve represents the concrete value added by making optimal, dynamic trading decisions.

### 5. The Policy Surface
```julia
# Plot 5: 3D Policy Surface
fig5 = plot_policy_surface(policies, params, t_check, state_vals, "r", asset_names)
```
**What it shows**: This generates a 3D surface plot visualizing the optimal portfolio weight on the Z-axis, with Wealth ($W$) on the X-axis and the state variable ($r$) on the Y-axis.
- **The Full Picture**: This provides a complete view of the agent's policy rule, showing exactly how their intertemporal hedging demands (driven by the Y-axis) interact simultaneously with their risk aversion (driven by the X-axis).