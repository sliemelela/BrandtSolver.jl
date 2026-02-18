module BrandtSolver

using LinearAlgebra
using Statistics
using StaticArrays
using ForwardDiff
using Interpolations
using Combinatorics
using NonlinearSolve
using Roots
using FinancialMarketSimulation

# --- Exports ---
export SolverParams, UtilityFunctions
export StandardOLS, TrimmedOLS
export DebugRecorder, NoOpRecorder
export create_utility_from_ad
export solve_portfolio_problem
export calculate_expected_utility

# --- Includes ---
include("types.jl")
include("utility.jl")
include("regression.jl")
include("interface.jl") # The bridge to SimulationWorld
include("core.jl")

end
