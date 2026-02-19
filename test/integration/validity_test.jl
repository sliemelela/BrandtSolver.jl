using Test
using BrandtSolver
using FinancialMarketSimulation
using LinearAlgebra
using Statistics
using StaticArrays
using Random
using Printf

include("../test_helpers.jl")

# ==============================================================================
# 1. SETUP
# ==============================================================================

function setup_merton_environment(; sims=10_000, T=5.0, M=5)
    # --- A. Define Physics (FinancialMarketSimulation) ---
    r_proc = VasicekProcess(:r, 0.3, 0.02, 0.001, 0.02, 1)

    σ_S = 0.20
    risk_premium = 0.05
    stock_drift(t, S, r_val) = S * (r_val + risk_premium)
    stock_diff(t, S, r_val)  = S * σ_S
    stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, 100.0, [2], [:r])
    re_proc = ExcessReturnProcess(:Re_Stock, :S, :r)

    config = MarketConfig(sims = sims, T = T, M = M, processes = [r_proc, stock_proc, re_proc])
    world = build_world(config)

    # --- B. Define Solver Config (BrandtSolver) ---
    # Notice we removed asset_names/state_names from SolverParams
    params = SolverParams(
        W_grid = [50.0, 75.0, 100.0, 125.0, 150.0],
        poly_order = 2,
        max_taylor_order = 4,
        trimming_α = 0.0
    )

    # --- C. Define Utility ---
    γ = 5.0
    crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
    utility = create_utility_from_ad(crra)

    return world, params, utility
end

function run_perturbation_test(Re_all, X_all, Y_all, params, policies, t_test, W_test, utility)
    println("\n  > Perturbation Check at t=$t_test, W=$W_test")

    pol_funcs = policies[t_test]
    sims = size(X_all, 1)

    ω_raw = [pol_funcs[i](W_test) for i in 1:sims]
    ω_star = mean(ω_raw)

    J_star, _ = calculate_expected_utility(Re_all, X_all, Y_all, policies, t_test, W_test, ω_raw, utility)
    println("    Baseline J(ω*): $J_star")

    ϵ = 0.05
    passed = true
    ω_up   = [w + SVector(ϵ) for w in ω_raw]
    ω_down = [w - SVector(ϵ) for w in ω_raw]

    J_up, _ = calculate_expected_utility(Re_all, X_all, Y_all, policies, t_test, W_test, ω_up, utility)
    J_down, _ = calculate_expected_utility(Re_all, X_all, Y_all, policies, t_test, W_test, ω_down, utility)

    if J_star > J_up && J_star > J_down
        println("    ✅ Peak confirmed: J(ω*) > J(±ϵ)")
    else
        println("    ❌ FAILED: Found better utility!")
        @printf("    J(ω*)=%.4f | J(+)=%.4f | J(-)=%.4f\n", J_star, J_up, J_down)
        passed = false
    end

    return passed
end

# ==============================================================================
# 2. TEST SUITE
# ==============================================================================

@testset "Full Pipeline Verification" begin

    println("\n--- 1. Setting up Merton Environment ---")
    Random.seed!(42)
    world, params, utility = setup_merton_environment(sims=20_000, T=5.0, M=5)

    # --- BRIDGE: Extract raw matrices for the solver ---
    Re_all = package_excess_returns(world, ["Re_Stock"])
    Z_all  = get_state_variables(world, ["r"])
    X_all, Y_all = create_risk_free_return_components(world, 0.0, nothing)

    println("--- 2. Running Solver ---")
    policies = solve_portfolio_problem(Re_all, Z_all, X_all, Y_all, params, utility)

    @testset "Merton Reproduction" begin
        target_weight = 0.25
        t_check = 1
        W_check = 100.0

        pol_funcs = policies[t_check]
        weights = [pol_funcs[i](W_check)[1] for i in 1:world.config.sims]
        avg_weight = mean(weights)

        println("  > Target Weight: $target_weight")
        println("  > Solver Weight: $avg_weight")

        @test isapprox(avg_weight, target_weight, atol=0.02)
    end

    @testset "Economic Validity (Perturbation)" begin
        passed = run_perturbation_test(
            Re_all, X_all, Y_all, params, policies, 1, 100.0, utility
        )
        @test passed
    end
end