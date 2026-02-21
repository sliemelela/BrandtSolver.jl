# BrandtSolver.jl

A high-performance Julia package for solving discrete-time dynamic portfolio choice problems
using the simulation-based method originally proposed by [Brandt_etal_2005](@cite).

In many realistic financial models (e.g., stochastic interest rates, predictability in
asset returns, non-tradable income), analytical solutions to portfolio choice problems are hard to come by.
`BrandtSolver.jl` overcomes this by combining already existing simulations of asset paths with cross-sectional regressions
to dynamically approximate the optimal portfolio policy backwards through time.

More specifically, in this package we consider portfolio choice problems at times
$t = 0, 1, 2, \ldots, T - 1$, where $T$ is some terminal date.
This portfolio choice problem at time $t$ is defined by an investor who maximizes the expected utility
of their wealth at the terminal date $T$ by trading $N$ risky assets and a risk-free asset (cash).
Formally the investor's problem at time $t$ is
```math
    V_t(W_t, Z_t)
    = \max_{\{\omega_s\}_{s = t}^{T - 1}} \mathbb{E}_t[u(W_T)]
```
subject to the sequence of budget constraints
```math
    W_{s + 1} = W_s (\omega_s^\top R^e_{s + 1} + R_{s + 1})
```
for all $s \geq t$.
Here $R^e_{s + 1}$ can be interpreted as the excess return of the risky assets over the risk-ree
asset, and $R_{s + 1}$ is the gross return of other processes that _may_ depend on wealth
$W_s$.
Furthermore, $\{\omega_s\}_{s=t}^{T - 1}$ is the sequence of portfolio weights chosen at times
$s = t, \ldots, T - 1$ and $u$ is the investor's utility function.
The process $Z_t$ is a vector of state variables that are relevant for the investor's decision making.
The goal of this package is to find $\{\omega_s\}_{s=0}^{T - 1}$



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

## Quickstart
ff

## References
```@bibliography
```