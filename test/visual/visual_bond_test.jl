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

println("\n--- Visual Verification: Bond + Stock Portfolio ---")


# ==============================================================================
# 1. SETUP (Stochastic Rates Market)
# ==============================================================================
function setup_bond_market()
    # 1. Interest Rate (Vasicek)
    # We need volatility here so the bond is actually risky!
    # r0 = 5%, Mean = 5%, Vol = 2%, Speed = 0.2
    r_proc = VasicekProcess(:r, 0.2, 0.05, 0.02, 0.05, 1)

    # 2. Stock (Correlated with Rate)
    # Drift 8%, Vol 20%
    stock_drift(t, S) = 0.08 * S
    stock_diff(t, S)  = 0.20 * S
    # Uses Shock 2
    stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, 100.0, [2])

    # 3. Bond (Zero Coupon)
    # Depends on :r. We define risk factors [ϕ_r, ϕ_S] to generate a risk premium.
    # Negative factors usually imply positive risk premium.
    market_risk = [-0.2, -0.1]
    bond_proc = NominalBondProcess(:P_N, r_proc, market_risk)

    # 4. Excess Returns (The Tradable Assets)
    re_stock = ExcessReturnProcess(:Re_Stock, :S, :r)
    re_bond  = ExcessReturnProcess(:Re_Bond, :P_N, :r)

    # 5. Config
    # Correlation between Rate (1) and Stock (2)
    ρ = [ 1.0 -0.4;
         -0.4  1.0]

    config = MarketConfig(
        sims = 2000,
        T = 5.0,
        M = 10, # Biannual decisions
        processes = [r_proc, stock_proc, bond_proc, re_stock, re_bond],
        correlations = ρ
    )

    world = build_world(config)

    # 6. Solver Params
    params = SolverParams(
        W_grid = [50.0, 100.0, 150.0, 200.0],
        poly_order = 2,
        max_taylor_order = 4,
        trimming_α = 0.01,
    )

    # 7. Utility
    γ = 5.0
    crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
    utility = create_utility_from_ad(crra)

    return world, params, utility
end

function setup_bond_market_merton_stock()

    # Market Parameters
    sims = 2000
    T = 5.0
    dt = 0.5
    ρ_rS = -0.4
    ρ = [ 1.0 ρ_rS;
         ρ_rS  1.0]

    # Market price of risk and factor loadings
    ϕ_r, ϕ_S = 0.075, -0.2
    mpr = [ϕ_r, ϕ_S]
    λ_S = -ϕ_r * ρ_rS - ϕ_S
    λ_r = -ϕ_r - ϕ_S * ρ_rS

    # Interest rate process (Vasicek)
    κ_r, θ_r, σ_r, r0 = 0.2, 0.05, 0.02, 0.05
    idx_r_shock = 1
    r_proc = VasicekProcess(:r, κ_r, θ_r, σ_r, r0, idx_r_shock)

    # Stock process (dS/S = (r_t + λσ)dt + σdW)
    σ_S = 0.2
    S_0 = 100.0
    idx_S_shock = 2
    stock_drift(t, S, r_val) = S * (r_val + λ_S * σ_S)
    stock_diff(t, S, r_val)  = S * σ_S
    stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, S_0, [idx_S_shock], [:r])

    # Nominal Bond Process
    bond_proc = NominalBondProcess(:P_N, r_proc, mpr)

    # Excess Returns
    re_stock = ExcessReturnProcess(:Re_Stock, :S, :r)
    re_bond  = ExcessReturnProcess(:Re_Bond, :P_N, :r)

    # Simulation of configuration
    config = MarketConfig(
        sims = sims,
        T = T,
        dt = dt,
        processes = [r_proc, stock_proc, bond_proc, re_stock, re_bond],
        correlations = ρ
    )
    world = build_world(config)

    # Solver Parameters
    params = SolverParams(
        W_grid = [100.0, 150.0, 200.0],
        poly_order = 2,
        max_taylor_order = 4,
        trimming_α = 0.01,
    )

    # Utility
    γ = 5.0
    crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
    utility = create_utility_from_ad(crra)

    return world, params, utility
end

@testset "Visual Bond Diagnostics" begin
    Random.seed!(999)

    println("  > Running Solver...")
    world, params, utility = setup_bond_market_merton_stock()

    asset_names = ["Re_Bond"]
    state_names = ["r"]

    # Extract matrices
    Re_all = package_excess_returns(world, asset_names)
    Z_all  = get_state_variables(world, state_names)
    X_all, Y_all = create_risk_free_return_components(world, 0.0, nothing)

    # Run decoupled solver
    policies = solve_portfolio_problem(Re_all, Z_all, X_all, Y_all, params, utility)

    println("  > Generating Plots...")

    t_check = 9
    fig1 = plot_policy_rules(policies, params, t_check, asset_names; samples=20)
    save(joinpath(OUTPUT_DIR, "bond_1_policy_vs_wealth.png"), fig1)

    state_vals = world.paths.r[:, t_check]
    fig2 = plot_state_dependence(policies, params, t_check, state_vals, "r", asset_names; fix_W=100.0)
    save(joinpath(OUTPUT_DIR, "bond_2_policy_vs_rate.png"), fig2)

    times = range(0, world.config.T, length=world.config.M+1)[1:end-1]
    fig3 = plot_realized_weights(Re_all, X_all, Y_all, times, policies, asset_names; W_init=100.0)
    save(joinpath(OUTPUT_DIR, "bond_3_realized_weights.png"), fig3)

    fig4 = plot_value_vs_utility(Re_all, X_all, Y_all, params, policies, utility; t_check=world.config.M)
    save(joinpath(OUTPUT_DIR, "bond_4_value_vs_utility.png"), fig4)

    fig5 = plot_policy_surface(policies, params, t_check, state_vals, "r", asset_names)
    save(joinpath(OUTPUT_DIR, "bond_5_policy_surface.png"), fig5)

    println("--- Visual Bond Test Complete ---")
end