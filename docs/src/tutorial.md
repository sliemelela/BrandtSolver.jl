# Tutorial: Solving a Dynamic Portfolio Choice Problem

In this tutorial, we will use `BrandtSolver.jl` alongside `FinancialMarketSimulation.jl` to solve a classic dynamic portfolio choice problem.
We will model an investor with Constant Relative Risk Aversion (CRRA) utility who must allocate their wealth between a risk-free asset with a stochastic interest rate and a risky stock.

Because the interest rate is stochastic and correlated with the stock, the investor will exhibit **intertemporal hedging demand**—their optimal allocation will depend on the current interest rate!

## Step 1: Simulating the Market
First, we need to generate the "paths" of the world. We use `FinancialMarketSimulation.jl` to build a market with a mean-reverting Vasicek interest rate and a stock whose drift depends on that rate.

```julia
using BrandtSolver
using FinancialMarketSimulation
using StaticArrays

# 1. Define the Interest Rate (Vasicek)
# Mean-reverting to 5%, with 3% volatility
r_proc = VasicekProcess(:r, 0.5, 0.05, 0.03, 0.05, 1)

# 2. Define the Stock
# 8% drift, 20% volatility
stock_drift(t, S) = 0.08 * S
stock_diff(t, S)  = 0.20 * S
stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, 100.0, [2])

# 3. Define the Excess Return Process
re_proc = ExcessReturnProcess(:Re_Stock, :S, :r)

# 4. Configure the Simulation
# We introduce a -0.5 correlation between the interest rate and the stock
ρ = [ 1.0 -0.5;
     -0.5  1.0]

config = MarketConfig(
    sims = 2000,
    T = 5.0,     # 5 years
    M = 10,      # 10 decision steps (biannual rebalancing)
    processes = [r_proc, stock_proc, re_proc],
    correlations = ρ
)

# Generate the simulated paths
world = build_world(config)
```

## Step 2: Configuring the Solver
Next, we define the parameters for our dynamic programming solver. We need to specify the wealth grid over which to evaluate the policy, the order of our Taylor expansions, and our utility function.

```julia
# 1. Setup Solver Parameters
params = SolverParams(
    W_grid = [50.0, 100.0, 150.0], # Wealth nodes to interpolate between
    poly_order = 2,                # Use up to Z^2 in regressions
    max_taylor_order = 4,          # 4th-order Taylor expansion of the value function
    trimming_α = 0.01              # Discard extreme 1% of paths to stabilize regressions
)

# 2. Define Utility using Automatic Differentiation
γ = 5.0
crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
utility = create_utility_from_ad(crra)
```


## Step 3: Formatting Data for the Solver
BrandtSolver.jl expects the simulation data in specific matrix formats. We extract these directly from our simulated world.

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

### Step 5: Visualizing the Results
`BrandtSolver.jl` comes with a suite of plotting tools (powered by Makie) to analyze the generated policies.

```julia
using CairoMakie

t_check = 9 # Look at the penultimate time step

# Plot 1: Policy as a function of Wealth
fig1 = plot_policy_rules(policies, params, t_check, asset_names; samples=20)

# Plot 2: Policy as a function of the State Variable (Interest Rate)
state_vals = world.paths.r[:, t_check]
fig2 = plot_state_dependence(policies, params, t_check, state_vals, "r", asset_names; fix_W=100.0)

# Plot 3: Forward simulation of the realized weights over time
times = range(0, world.config.T, length=world.config.M+1)[1:end-1]
fig3 = plot_realized_weights(Re_all, X_all, Y_all, times, policies, asset_names; W_init=100.0)
```