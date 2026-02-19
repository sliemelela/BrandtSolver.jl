using Test
using BrandtSolver
using FinancialMarketSimulation
using CairoMakie
using LinearAlgebra
using Statistics
using StaticArrays
using Random

# Ensure the output directory exists
OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
if !isdir(OUTPUT_DIR)
    mkdir(OUTPUT_DIR)
end

println("\n--- Visual Verification Suite ---")
println("  > Plots will be saved to: $OUTPUT_DIR")

# ==============================================================================
# 1. SETUP (Fast Simulation)
# ==============================================================================
function setup_fast_market()
    # 1. Interest Rate (Vasicek)
    # Mean reverting to 5%, high volatility to make the plots interesting
    r_proc = VasicekProcess(:r, 0.5, 0.05, 0.03, 0.05, 1)

    # 2. Stock (Correlated with Rate)
    # High vol (20%), Correlation with rates = -0.5
    # This ensures the "State Dependency" plot shows a pattern.
    stock_drift(t, S) = 0.08 * S # 8% drift
    stock_diff(t, S)  = 0.20 * S # 20% vol
    stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, 100.0, [2])

    # 3. Returns
    re_proc = ExcessReturnProcess(:Re_Stock, :S, :r)

    # 4. Config
    ρ = [1.0 -0.5; -0.5 1.0] # Correlation

    config = MarketConfig(
        sims = 1000, # Enough for clean plots, fast enough for tests
        T = 5.0,
        M = 10,      # 10 steps (biannual rebalancing)
        processes = [r_proc, stock_proc, re_proc],
        correlations = ρ
    )

    world = build_world(config)

    # 5. Solver Params
    params = SolverParams(
        asset_names = ["Re_Stock"],
        state_names = ["r"],
        W_grid = [50.0, 100.0, 150.0], # Coarse grid is fine for visual check
        poly_order = 2,
        max_taylor_order = 4,
        p_income = 0.0,
        O_t_real_path = nothing,
        trimming_α = 0.01,
        γ = 5.0
    )

    # 6. Utility
    γ = params.γ
    crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
    utility = create_utility_from_ad(crra)

    return world, params, utility
end

function setup_fast_merton_market()
    # 1. Interest Rate (Effective Constant: r = 2%)
    # Very low volatility (0.001) to mimic constant rate
    r_proc = VasicekProcess(:r, 0.3, 0.02, 0.001, 0.02, 1)

    # 2. Stock Process (GBM)
    # Target: dS/S = (r + λσ)dt + σdW
    # Risk Premium = 5% (0.05), Vol = 20% (0.20) -> Implied Sharpe = 0.25
    σ_S = 0.20
    risk_premium = 0.05

    stock_drift(t, S, r_val) = S * (r_val + risk_premium)
    stock_diff(t, S, r_val)  = S * σ_S

    # Stock depends on :r and uses shock 2
    stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, 100.0, [2], [:r])

    # 3. Excess Return
    re_proc = ExcessReturnProcess(:Re_Stock, :S, :r)

    # 4. Build Config
    # No correlations -> Identity matrix (uncorrelated shocks)
    config = MarketConfig(
        sims = 2000, # 2k sims is enough for clean plots but fast to run
        T = 5.0,
        M = 10,      # 10 steps (biannual rebalancing)
        processes = [r_proc, stock_proc, re_proc]
    )

    world = build_world(config)

    # 5. Solver Params
    params = SolverParams(
        asset_names = ["Re_Stock"],
        state_names = ["r"],
        W_grid = [50.0, 75.0, 100.0, 125.0, 150.0],
        poly_order = 2,
        max_taylor_order = 4,
        p_income = 0.0,
        O_t_real_path = nothing,
        trimming_α = 0.0, # No trimming needed for Merton
        γ = 5.0
    )

    # 6. Utility
    γ = params.γ
    crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
    utility = create_utility_from_ad(crra)

    return world, params, utility
end

@testset "Visual Diagnostics" begin
    Random.seed!(42)

    # 1. Run Solver
    println("  > Running Solver...")
    # world, params, utility = setup_fast_merton_market()
    world, params, utility = setup_fast_market()
    policies = solve_portfolio_problem(world, params, utility)

    @test length(policies) == world.config.M + 1

    # 2. Generate Plots
    println("  > Generating Plots...")

    # Plot A: Policy Rules (The Brain)
    # Check decision at t=1
    fig1 = plot_policy_rules(policies, params, 10; samples=20)
    save(joinpath(OUTPUT_DIR, "1_policy_vs_wealth.png"), fig1)
    @test isfile(joinpath(OUTPUT_DIR, "1_policy_vs_wealth.png"))
    println("    - Saved Policy vs Wealth")

    # Plot B: State Dependency (The Context)
    # Check how weights depend on 'r' at t=1, W=100
    fig2 = plot_state_dependence(policies, world, params, 10, :r; fix_W=100.0)
    save(joinpath(OUTPUT_DIR, "2_policy_vs_rate.png"), fig2)
    @test isfile(joinpath(OUTPUT_DIR, "2_policy_vs_rate.png"))
    println("    - Saved Policy vs Rate")

    # Plot C: Realized Paths (The Result)
    # Re-simulate to see actual portfolio behavior
    fig3 = plot_realized_weights(world, params, policies; W_init=100.0)
    save(joinpath(OUTPUT_DIR, "3_realized_weights.png"), fig3)
    @test isfile(joinpath(OUTPUT_DIR, "3_realized_weights.png"))
    println("    - Saved Realized Weights")

    # Plot D: Value vs Utility (The Sanity Check)
    # Check at the last step (M)
    fig4 = plot_value_vs_utility(world, params, policies, utility; t_check=world.config.M)
    save(joinpath(OUTPUT_DIR, "4_value_vs_utility.png"), fig4)
    @test isfile(joinpath(OUTPUT_DIR, "4_value_vs_utility.png"))
    println("    - Saved Value vs Utility")

    # Plot E: Surface (The Big Picture)
    fig5 = plot_policy_surface(policies, world, params, 10, :r)
    save(joinpath(OUTPUT_DIR, "5_policy_surface.png"), fig5)
    @test isfile(joinpath(OUTPUT_DIR, "5_policy_surface.png"))
    println("    - Saved Policy Surface")

    println("--- Visual Test Complete (Check /test/output/) ---")
end