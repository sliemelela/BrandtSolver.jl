using Test
using BrandtSolver
using FinancialMarketSimulation
using CairoMakie
using LinearAlgebra
using Statistics
using StaticArrays
using Random

include("../test_helpers.jl")

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
        W_grid = [50.0, 100.0, 150.0], # Coarse grid is fine for visual check
        poly_order = 2,
        max_taylor_order = 4,
        trimming_α = 0.01,
    )

    # 6. Utility
    γ = 5.0
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
        W_grid = [50.0, 75.0, 100.0, 125.0, 150.0],
        poly_order = 2,
        max_taylor_order = 4,
        trimming_α = 0.0, # No trimming needed for Merton
    )

    # 6. Utility
    γ = 5.0
    crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
    utility = create_utility_from_ad(crra)

    return world, params, utility
end

@testset "Visual Diagnostics" begin
    Random.seed!(42)
    println("  > Running Solver...")
    world, params, utility = setup_fast_market()

    asset_names = ["Re_Stock"]
    state_names = ["r"]

    # Extract matrices
    Re_all = package_excess_returns(world, asset_names)
    Z_all  = get_state_variables(world, state_names)
    X_all, Y_all = create_risk_free_return_components(world, 0.0, nothing)

    # Run decoupled solver
    policies = solve_portfolio_problem(Re_all, Z_all, X_all, Y_all, params, utility)

    @test length(policies) == world.config.M + 1

    println("  > Generating Plots...")

    fig1 = plot_policy_rules(policies, params, 10, asset_names; samples=20)
    save(joinpath(OUTPUT_DIR, "1_policy_vs_wealth.png"), fig1)

    state_vals = world.paths.r[:, 10]
    fig2 = plot_state_dependence(policies, params, 10, state_vals, "r", asset_names; fix_W=100.0)
    save(joinpath(OUTPUT_DIR, "2_policy_vs_rate.png"), fig2)

    times = range(0, world.config.T, length=world.config.M+1)[1:end-1]
    fig3 = plot_realized_weights(Re_all, X_all, Y_all, times, policies, asset_names; W_init=100.0)
    save(joinpath(OUTPUT_DIR, "3_realized_weights.png"), fig3)

    fig4 = plot_value_vs_utility(Re_all, X_all, Y_all, params, policies, utility; t_check=world.config.M)
    save(joinpath(OUTPUT_DIR, "4_value_vs_utility.png"), fig4)

    fig5 = plot_policy_surface(policies, params, 10, state_vals, "r", asset_names)
    save(joinpath(OUTPUT_DIR, "5_policy_surface.png"), fig5)

    println("--- Visual Test Complete ---")
end