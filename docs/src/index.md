# BrandtSolver.jl

A high-performance Julia package for solving discrete-time dynamic portfolio choice problems
using the simulation-based method originally proposed by [Brandt_etal_2005](@cite).

In many realistic financial models (e.g., stochastic interest rates, predictability in
asset returns, non-tradable income), analytical solutions to portfolio choice problems are hard to come by.
`BrandtSolver.jl` overcomes this by combining already existing simulations of asset paths with cross-sectional regressions
to dynamically approximate the optimal portfolio policy backwards through time.




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