using Test
using BrandtSolver
using FinancialMarketSimulation
using LinearAlgebra
using Statistics
using StaticArrays
using Random
using Printf

# ==============================================================================
# 1. HELPERS
# ==============================================================================

"""
Sets up a SimulationWorld and SolverParams that mimic the Merton problem:
- Constant Interest Rate (Vasicek with low vol)
- Geometric Brownian Motion Stock (GenericSDE)
- Constant Risk Premium
"""
function setup_merton_environment(; sims=10_000, T=5.0, M=5)

    # --- A. Define Physics (FinancialMarketSimulation) ---

    # 1. Interest Rate (Effective Constant: r = 2%)
    r_proc = VasicekProcess(:r, 0.3, 0.02, 0.001, 0.02, 1)

    # 2. Stock Process (GBM)
    # Target: dS/S = (r + λσ)dt + σdW
    # We want Risk Premium = 5% (0.05) and Vol = 20% (0.20)
    # Implied Sharpe λ = 0.25
    σ_S = 0.20
    risk_premium = 0.05

    stock_drift(t, S, r_val) = S * (r_val + risk_premium)
    stock_diff(t, S, r_val)  = S * σ_S

    # Stock depends on :r and uses shock 2
    stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, 100.0, [2], [:r])

    # 3. Excess Return
    re_proc = ExcessReturnProcess(:Re_Stock, :S, :r)

    # 4. Build Config
    config = MarketConfig(
        sims = sims,
        T = T,
        M = M,
        processes = [r_proc, stock_proc, re_proc]
    )

    world = build_world(config)

    # --- B. Define Solver Config (BrandtSolver) ---

    params = SolverParams(
        asset_names = ["Re_Stock"],
        state_names = ["r"],
        W_grid = [50.0, 75.0, 100.0, 125.0, 150.0],
        poly_order = 2,
        max_taylor_order = 4,
        trimming_α = 0.0,
        γ = 5.0
    )

    # --- C. Define Utility ---
    γ = params.γ
    crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
    utility = create_utility_from_ad(crra)

    return world, params, utility
end

"""
Checks if the solver's policy is a local maximum by perturbing it.
"""
function run_perturbation_test(world, params, policies, t_test, W_test, utility)
    println("\n  > Perturbation Check at t=$t_test, W=$W_test")

    # 1. Get the Solver's Optimal Policy (Average across sims)
    pol_funcs = policies[t_test]
    sims = world.config.sims

    # We evaluate the policy for every simulation path to get the "suggested" vector
    ω_raw = [pol_funcs[i](W_test) for i in 1:sims]
    ω_star = mean(ω_raw) # The "mean optimal" weight

    # 2. Calculate Baseline Utility J(ω*)
    # We force the simulation to use ω_raw for the next step, then use policies thereafter
    J_star, _ = calculate_expected_utility(
        world, params, policies, t_test, W_test, ω_raw, utility
    )
    println("    Baseline J(ω*): $J_star")

    # 3. Perturb and Compare
    ϵ = 0.05 # 5% shift
    passed = true

    # Create perturbed policies (uniformly shifted for all sims)
    # ω_up = ω_raw + ϵ
    ω_up   = [w + SVector(ϵ) for w in ω_raw]
    ω_down = [w - SVector(ϵ) for w in ω_raw]

    J_up, _ = calculate_expected_utility(
        world, params, policies, t_test, W_test, ω_up, utility
    )

    J_down, _ = calculate_expected_utility(
        world, params, policies, t_test, W_test, ω_down, utility
    )

    # 4. Assert Concavity (Star should be higher than neighbors)
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
    # Use high sims for accuracy
    Random.seed!(42)
    world, params, utility = setup_merton_environment(sims=20_000, T=5.0, M=5)

    println("--- 2. Running Solver ---")
    # This runs the backward recursion
    policies = solve_portfolio_problem(world, params, utility)

    @testset "Merton Reproduction" begin
        # Theoretical Answer:
        # Weight = (μ - r) / (γ * σ^2)
        #        = 0.05 / (5.0 * 0.20^2) = 0.05 / 0.20 = 0.25
        target_weight = 0.25

        # Check policy at t=1 (first decision point)
        t_check = 1
        W_check = 100.0

        # Extract weights from all simulations
        pol_funcs = policies[t_check]
        weights = [pol_funcs[i](W_check)[1] for i in 1:world.config.sims]
        avg_weight = mean(weights)

        println("  > Target Weight: $target_weight")
        println("  > Solver Weight: $avg_weight")

        # Allow 5% relative error due to Monte Carlo noise
        @test isapprox(avg_weight, target_weight, atol=0.02)
    end

    @testset "Economic Validity (Perturbation)" begin
        # Verify that the solution is a true local maximum
        passed = run_perturbation_test(
            world, params, policies, 1, 100.0, utility
        )
        @test passed
    end
end