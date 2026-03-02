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
    # stock_drift(t, S, r_val) = S * (0.8)
    stock_diff(t, S, r_val)  = S * σ_S
    stock_proc = GenericSDEProcess(:S, stock_drift, stock_diff, S_0, [idx_S_shock], [:r])

    # Nominal Bond Process
    # T_mat = T + 5.0
    T_mat = T
    bond_proc = NominalBondProcess(:P_N, r_proc, T_mat, mpr)

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

    asset_names = ["Re_Stock", "Re_Bond"]
    state_names = ["r"]

    # Extract matrices
    Re_all = package_excess_returns(world, asset_names)
    Z_all  = get_state_variables(world, state_names)
    X_all, Y_all = create_risk_free_return_components(world, 0.0, nothing)

    # Run decoupled solver
    policies = solve_portfolio_problem(Re_all, Z_all, X_all, Y_all, params, utility)

    println("  > Generating Plots...")

    t_check = world.config.M
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



    function plot_objective_surface(world, params, utility, policies, t_check, W_test)
        # 1. Extract Data
        asset_names = ["Re_Stock", "Re_Bond"]
        Re_all = package_excess_returns(world, asset_names)
        X_all, Y_all = create_risk_free_return_components(world, 0.0, nothing)

        # 2. Define Weight Grids (Adjust bounds based on your expected solution)
        w_stock_grid = range(0.0, 2.0, length=25)
        w_bond_grid = range(-1.0, 2.0, length=25)

        CE_surface = zeros(length(w_stock_grid), length(w_bond_grid))

        println("Evaluating true objective surface (this might take a minute)...")
        for (i, w_s) in enumerate(w_stock_grid)
            for (j, w_b) in enumerate(w_bond_grid)
                # Force this specific portfolio allocation at t_check
                w_force = SVector{2, Float64}(w_s, w_b)

                _, CE = calculate_expected_utility(
                    Re_all, X_all, Y_all, policies, t_check, W_test, w_force, utility
                )
                CE_surface[i, j] = CE
            end
        end

        # 3. Get the point the solver ACTUALLY chose for the average path at W_test
        # We take the mean policy across all simulations for this specific wealth
        sims = size(X_all, 1)
        solver_weights = mean([policies[t_check][s](W_test) for s in 1:sims])

        # Evaluate the CE at the solver's chosen point
        _, solver_CE = calculate_expected_utility(
            Re_all, X_all, Y_all, policies, t_check, W_test, solver_weights, utility
        )

        # 4. Plotting
        fig = Figure(size = (800, 600))
        ax = Axis3(fig[1, 1],
            title = "Objective Surface (Certainty Equivalent) at t=$t_check, W=$W_test",
            xlabel = "Stock Weight (ω_S)",
            ylabel = "Bond Weight (ω_B)",
            zlabel = "Certainty Equivalent"
        )

        # Plot the surface
        surface!(ax, w_stock_grid, w_bond_grid, CE_surface, colormap=:viridis, alpha=0.8)

        # Plot the point chosen by the Brandt Solver
        scatter!(ax, [solver_weights[1]], [solver_weights[2]], [solver_CE],
            color=:red, markersize=15, label="Solver Choice"
        )

        axislegend(ax)
        return fig
    end

    # Usage: Run this after your solver finishes in your test script
    fig_surface = plot_objective_surface(world, params, utility, policies, t_check, 1.0)
    save(joinpath(OUTPUT_DIR, "objective_surface.png"), fig_surface)

    println("--- Visual Bond Test Complete ---")
end