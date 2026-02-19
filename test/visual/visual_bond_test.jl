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

println("\n--- Visual Verification: Bond + Stock Portfolio ---")
println("  > Plots will be saved to: $OUTPUT_DIR")

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
        asset_names = ["Re_Stock", "Re_Bond"], # <--- Two Assets now!
        state_names = ["r"],                   # State variable
        W_grid = [50.0, 100.0, 150.0, 200.0],
        poly_order = 2,
        max_taylor_order = 4,
        trimming_α = 0.01,
        γ = 5.0
    )

    # 7. Utility
    γ = params.γ
    crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
    utility = create_utility_from_ad(crra)

    return world, params, utility
end

function setup_bond_market_merton_stock()
    # 1. Interest Rate (Vasicek)
    # Volatile enough to make the bond risky and hedging interesting.
    # Mean 5%, Vol 2%, Speed 0.2
    r_proc = VasicekProcess(:r, 0.2, 0.05, 0.02, 0.05, 1)

    # 2. Stock (Merton-Style Drift)
    # Target: dS/S = (r_t + λσ)dt + σdW
    # Excess Return = (r_t + λσ) - r_t = λσ (Constant!)
    # This isolates Hedging Demand from Speculative Demand.
    σ_S = 0.20
    risk_premium = 0.05

    # Note: We now pass r_val into the drift function
    stock_drift(t, S, r_val) = S * (r_val + risk_premium)
    stock_diff(t, S, r_val)  = S * σ_S

    # IMPORTANT: We add [:r] to dependencies so the process reads the rate path
    stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, 100.0, [2], [:r])

    # 3. Bond (Zero Coupon)
    # Price depends on r. Returns will be volatile.
    market_risk = [-0.2, -0.1]
    bond_proc = NominalBondProcess(:P_N, r_proc, market_risk)

    # 4. Excess Returns
    re_stock = ExcessReturnProcess(:Re_Stock, :S, :r)
    re_bond  = ExcessReturnProcess(:Re_Bond, :P_N, :r)

    # 5. Config
    # Correlation between Rate (1) and Stock (2)
    ρ = [ 1.0 -0.4;
         -0.4  1.0]

    config = MarketConfig(
        sims = 2000,
        T = 5.0,
        M = 10,
        processes = [r_proc, stock_proc, bond_proc, re_stock, re_bond],
        correlations = ρ
    )

    world = build_world(config)

    # 6. Solver Params
    params = SolverParams(
        asset_names = ["Re_Stock", "Re_Bond"],
        state_names = ["r"],
        W_grid = [50.0, 100.0, 150.0, 200.0],
        poly_order = 2,
        max_taylor_order = 4,
        trimming_α = 0.01,
        γ = 5.0
    )

    # 7. Utility
    γ = params.γ
    crra(W) = (W^(1.0 - γ)) / (1.0 - γ)
    utility = create_utility_from_ad(crra)

    return world, params, utility
end

@testset "Visual Bond Diagnostics" begin
    Random.seed!(999)

    # 1. Run Solver
    println("  > Running Solver...")
    # world, params, utility = setup_bond_market()
    world, params, utility = setup_bond_market_merton_stock()
    policies = solve_portfolio_problem(world, params, utility)

    # 2. Generate Plots
    println("  > Generating Plots...")

    # Plot A: Policy Rules (The Brain)
    # EXPECTATION: You should see 2 panels (Stock, Bond).
    # Bond weights might slope with Wealth due to intertemporal hedging.
    fig1 = plot_policy_rules(policies, params, 1; samples=20)
    save(joinpath(OUTPUT_DIR, "bond_1_policy_vs_wealth.png"), fig1)
    println("    - Saved Policy vs Wealth")

    # Plot B: State Dependency (The Context)
    # EXPECTATION: Strong relationship between Bond Weight and Interest Rate (r)
    # because r drives the bond's volatility and drift directly.
    fig2 = plot_state_dependence(policies, world, params, 1, :r; fix_W=100.0)
    save(joinpath(OUTPUT_DIR, "bond_2_policy_vs_rate.png"), fig2)
    println("    - Saved Policy vs Rate")

    # Plot C: Realized Paths (The Result)
    # EXPECTATION: Two distinct weight trajectories fluctuating over time.
    fig3 = plot_realized_weights(world, params, policies; W_init=100.0)
    save(joinpath(OUTPUT_DIR, "bond_3_realized_weights.png"), fig3)
    println("    - Saved Realized Weights")

    # Plot D: Value vs Utility (The Sanity Check)
    fig4 = plot_value_vs_utility(world, params, policies, utility; t_check=world.config.M)
    save(joinpath(OUTPUT_DIR, "bond_4_value_vs_utility.png"), fig4)
    println("    - Saved Value vs Utility")

    # Plot E: Surface (The Big Picture)
    # 3D view of how weights change with W and r
    fig5 = plot_policy_surface(policies, world, params, 1, :r)
    save(joinpath(OUTPUT_DIR, "bond_5_policy_surface.png"), fig5)
    println("    - Saved Policy Surface")

    println("--- Visual Bond Test Complete (Check /test/output/) ---")
end