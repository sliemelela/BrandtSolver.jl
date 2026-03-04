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

function setup_bond_market_merton_stock()

    # Market Parameters
    sims = 2000
    T = 0.06
    dt = 0.02
    ρ_rS = -0.4
    ρ_rπ = 0.3
    ρ_πS = 0.2
    ρ = [ 1.0 ρ_rπ ρ_rS;
         ρ_rπ  1.0 ρ_πS;
         ρ_rS  ρ_πS 1.0]

    # Market price of risk and factor loadings
    ϕ_r, ϕ_S = 0.075, -0.2
    mpr = [ϕ_r, ϕ_S]
    λ_S = -ϕ_r * ρ_rS - ϕ_S
    λ_r = -ϕ_r - ϕ_S * ρ_rS

    # Interest rate process (Vasicek)
    κ_r, θ_r, σ_r, r0 = 0.2, 0.05, 0.02, 0.05
    κ_π, θ_π, σ_π, π0 = 0.2, 0.02, 0.01, 0.02
    idx_r_shock = 1
    idx_π_shock = 2
    r_proc = VasicekProcess(:r, κ_r, θ_r, σ_r, r0, idx_r_shock)
    π_proc = VasicekProcess(:π, κ_π, θ_π, σ_π, π0, idx_π_shock)

    # CPI process (for Real Bond)
    cpi_drift(t, Pi, pi_val) = Pi * pi_val
    cpi_diff(t, Pi, pi_val)  = 0.0
    cpi_model = GenericSDEProcess(
        :CPI, cpi_drift, cpi_diff, 100.0,
        Int[],   # No Shocks
        [:π]    # Dependency: Reads the :pi path
    )

    # Stock process (dS/S = (r_t + λσ)dt + σdW)
    σ_S = 0.2
    S_0 = 100.0
    idx_S_shock = 3
    stock_drift(t, S, r_val) = S * (r_val + λ_S * σ_S)
    stock_diff(t, S, r_val)  = S * σ_S
    stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, S_0, [idx_S_shock], [:r])

    # Nominal Bond Process
    T_mat = T + 5.0
    bond_proc = NominalBondProcess(:P_N, r_proc, T_mat, mpr)

    # Inflation Bond Process
    infl_bond = InflationBondProcess(:P_I, r_proc, π_proc, :CPI, T_mat, mpr)

    # Excess Returns
    re_stock = ExcessReturnProcess(:Re_Stock, :S, :r)
    re_bond  = ExcessReturnProcess(:Re_Bond, :P_N, :r)
    re_infl_bond = ExcessReturnProcess(:Re_Infl_Bond, :P_I, :r)

    # Simulation of configuration
    config = MarketConfig(
        sims = sims,
        T = T,
        dt = dt,
        processes = [r_proc, π_proc, cpi_model, stock_proc, bond_proc, infl_bond, re_stock, re_bond, re_infl_bond],
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

    asset_names = ["Re_Stock", "Re_Bond", "Re_Infl_Bond"]
    state_names = ["r", "π", "S"]

    # Extract matrices
    Re_all = package_excess_returns(world, asset_names)
    Z_all  = get_state_variables(world, state_names)
    X_all, Y_all = create_risk_free_return_components(world, 0.0, nothing)

    # Run decoupled solver
    policies = solve_portfolio_problem(Re_all, Z_all, X_all, Y_all, params, utility)

    println("  > Generating Plots...")

    t_check = world.config.M
    fig1 = plot_policy_rules(policies, params, t_check, asset_names; samples=20)
    save(joinpath(OUTPUT_DIR, "infl_bond_1_policy_vs_wealth.png"), fig1)

    state_vals = world.paths.r[:, t_check]
    fig2 = plot_state_dependence(policies, params, t_check, state_vals, "r", asset_names; fix_W=100.0)
    save(joinpath(OUTPUT_DIR, "infl_bond_2_policy_vs_rate.png"), fig2)

    times = range(0, world.config.T, length=world.config.M+1)[1:end-1]
    fig3 = plot_realized_weights(Re_all, X_all, Y_all, times, policies, asset_names; W_init=100.0)
    save(joinpath(OUTPUT_DIR, "infl_bond_3_realized_weights.png"), fig3)

    fig4 = plot_value_vs_utility(Re_all, X_all, Y_all, params, policies, utility; t_check=world.config.M)
    save(joinpath(OUTPUT_DIR, "infl_bond_4_value_vs_utility.png"), fig4)

    fig5 = plot_policy_surface(policies, params, t_check, state_vals, "r", asset_names)
    save(joinpath(OUTPUT_DIR, "infl_bond_5_policy_surface.png"), fig5)

    println("--- Visual Inflation Bond Test Complete ---")
end