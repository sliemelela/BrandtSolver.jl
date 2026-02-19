module BrandtSolver

using LinearAlgebra
using Statistics
using StaticArrays
using ForwardDiff
using Interpolations
using Combinatorics
using NonlinearSolve
using Roots
using CairoMakie

# Core Exports
export SolverParams, UtilityFunctions
export StandardOLS, TrimmedOLS
export DebugRecorder, NoOpRecorder
export create_utility_from_ad
export solve_portfolio_problem
export calculate_expected_utility

# Plotting Exports
export plot_policy_rules, plot_state_dependence, plot_realized_weights
export plot_value_vs_utility, plot_policy_surface

# Includes
include("types.jl")
include("utility.jl")
include("regression.jl")
include("core.jl")
include("plotting.jl")

end