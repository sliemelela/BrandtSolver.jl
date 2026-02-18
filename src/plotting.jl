using CairoMakie
using Statistics
using StaticArrays
using FinancialMarketSimulation

# ==============================================================================
# 1. VISUALIZE DECISION RULES (Policy vs Wealth)
# ==============================================================================
"""
    plot_policy_rules(future_policies, solver_params, t_idx; samples=50)

Visualizes the optimal policy function ω(W) at a specific time `t_idx`.
Since the policy depends on the market state (which varies by simulation),
this plots multiple lines (one for each sampled simulation context) to show the spread.

# Arguments
- `t_idx`: Time index to inspect (e.g., 1 for the first decision).
- `samples`: Number of random simulation paths to plot (limits clutter).
"""
function plot_policy_rules(future_policies, solver_params, t_idx; samples=50)
    (; asset_names, W_grid) = solver_params

    # 1. Select Policies
    policies_at_t = future_policies[t_idx] # Vector of functions (one per sim)
    total_sims = length(policies_at_t)
    indices = rand(1:total_sims, min(samples, total_sims))

    n_assets = length(asset_names)

    # 2. Setup Figure
    fig = Figure(size = (1000, 400 * n_assets))

    for (k, asset) in enumerate(asset_names)
        ax = Axis(fig[k, 1],
            title = "Optimal Allocation to $asset (t=$t_idx)",
            xlabel = "Wealth (W)",
            ylabel = "Weight (ω)",
            titlesize = 16
        )

        # 3. Evaluate and Plot for each sample simulation
        for i in indices
            # Evaluate policy function for simulation i across the grid
            weights = [policies_at_t[i](w)[k] for w in W_grid]
            lines!(ax, W_grid, weights, color=(:blue, 0.3), linewidth=1.5)
        end

        # 4. Plot Mean Policy (Thick Line)
        mean_weights = zeros(length(W_grid))
        for (jw, w) in enumerate(W_grid)
            # Average across ALL simulations, not just samples
            vals = [policies_at_t[i](w)[k] for i in 1:total_sims]
            mean_weights[jw] = mean(vals)
        end
        lines!(ax, W_grid, mean_weights, color=:red, linewidth=3, label="Mean Policy")
        axislegend(ax)
    end

    return fig
end

# ==============================================================================
# 2. VISUALIZE STATE DEPENDENCY (Policy vs Market Variable)
# ==============================================================================
"""
    plot_state_dependence(future_policies, world, solver_params, t_idx, state_name; fix_W=100.0)

Visualizes how the policy changes with respect to a market state variable (e.g., Interest Rate).
It fixes Wealth at `fix_W` and scatters the policy weight against the state variable for all simulations.
"""
function plot_state_dependence(future_policies, world, solver_params, t_idx, state_name::Symbol; fix_W=100.0)
    (; asset_names) = solver_params

    # 1. Get State Data (X-axis)
    # Extract the column at time t_idx for the requested state variable
    if !hasproperty(world.paths, state_name)
        error("State variable :$state_name not found in world paths.")
    end
    state_values = getproperty(world.paths, state_name)[:, t_idx]

    # 2. Get Policy Data (Y-axis)
    policies_at_t = future_policies[t_idx]
    sims = length(policies_at_t)
    n_assets = length(asset_names)

    weights_matrix = zeros(sims, n_assets)
    for i in 1:sims
        w_vec = policies_at_t[i](fix_W)
        weights_matrix[i, :] = w_vec
    end

    # 3. Plot
    fig = Figure(size = (1000, 400 * n_assets))

    for (k, asset) in enumerate(asset_names)
        ax = Axis(fig[k, 1],
            title = "Sensitivity of $asset to $state_name (at W=$fix_W, t=$t_idx)",
            xlabel = String(state_name),
            ylabel = "Weight (ω)",
            titlesize = 16
        )

        scatter!(ax, state_values, weights_matrix[:, k],
                 color=(:black, 0.1), markersize=4)

        # Add a trend line (simple polynomial fit for visualization)
        # Using simple sorting for line plot
        p = sortperm(state_values)
        lines!(ax, state_values[p], weights_matrix[p, k], color=:red, linewidth=2)
    end

    return fig
end

# ==============================================================================
# 3. VISUALIZE REALIZED PATHS (Weight vs Time)
# ==============================================================================
# ==============================================================================
# 3. VISUALIZE REALIZED PATHS (Weight vs Time)
# ==============================================================================
"""
    plot_realized_weights(world, solver_params, future_policies; W_init=100.0)

Re-simulates the wealth trajectory using the optimal policies and plots the
actual portfolio weights realized over time. Handles NaNs (bankruptcies) robustly.
"""
function plot_realized_weights(world, solver_params, future_policies; W_init=100.0)
    (; asset_names, p_income, O_t_real_path) = solver_params

    # 1. Setup Simulation (Fast Re-run to capture weights)
    sim = world.config.sims
    T_steps = world.config.M + 1

    # Prepare Data
    if O_t_real_path === nothing
        O_t = zeros(sim, T_steps)
    else
        O_t = getproperty(world.paths, Symbol(O_t_real_path))
    end
    X, Y = create_risk_free_return_components(world, p_income, O_t)
    Re = package_excess_returns(world, asset_names)

    # 2. Step-by-step Loop to capture weights
    n_assets = length(asset_names)

    # Store weights: [sim, time, asset]
    weight_history = zeros(sim, T_steps-1, n_assets)
    W_current = fill(W_init, sim)

    for t in 1:(T_steps - 1)
        policies = future_policies[t]

        # Calculate weights for this step
        for i in 1:sim
            if isnan(W_current[i]) || W_current[i] <= 0
                weight_history[i, t, :] .= NaN # Mark as dead/invalid
                continue
            end

            ω = policies[i](W_current[i]) # Evaluate policy
            weight_history[i, t, :] = ω
        end

        # Advance Wealth
        r_free = X[:, t] .+ Y[:, t] ./ W_current
        re_next = Re[:, t+1]

        for i in 1:sim
            if isnan(W_current[i]); continue; end

            port_ret = dot(weight_history[i, t, :], re_next[i])
            W_next = W_current[i] * (r_free[i] + port_ret)

            # Catch bankruptcy or explosion
            if W_next <= 1e-9 || isnan(W_next)
                W_current[i] = NaN
            else
                W_current[i] = W_next
            end
        end
    end

    # 3. Plotting
    fig = Figure(size = (1000, 400 * n_assets))
    times = range(0, world.config.T, length=T_steps)[1:end-1] # Weights exist for t=1..M

    for (k, asset) in enumerate(asset_names)
        ax = Axis(fig[k, 1],
            title = "Realized Weight Path: $asset",
            xlabel = "Time",
            ylabel = "Weight",
            titlesize = 16
        )

        # Plot sample paths (Makie automatically breaks lines at NaNs)
        indices = rand(1:sim, min(50, sim))
        for i in indices
            lines!(ax, times, weight_history[i, :, k], color=(:blue, 0.15))
        end

        # Calculate Robust Mean & CI (Skipping NaNs)
        means = Float64[]
        lows = Float64[]
        highs = Float64[]

        for t in 1:length(times)
            data_t = weight_history[:, t, k]
            valid_data = filter(!isnan, data_t)

            if isempty(valid_data)
                push!(means, NaN); push!(lows, NaN); push!(highs, NaN)
            else
                push!(means, mean(valid_data))
                push!(lows, quantile(valid_data, 0.05))
                push!(highs, quantile(valid_data, 0.95))
            end
        end

        # Plot Statistics
        lines!(ax, times, means, color=:black, linewidth=2, label="Mean Path")
        band!(ax, times, lows, highs, color=(:black, 0.1), label="90% CI")

        axislegend(ax)
    end

    return fig
end


# ==============================================================================
# 4. DIAGNOSTIC: VALUE FUNCTION VS UTILITY
# ==============================================================================
"""
    plot_value_vs_utility(world, solver_params, future_policies, utility_struct; t_check=nothing)

Plots the approximated Value Function J(W) (computed via simulation) against the
theoretical Utility Function U(W) at a specific time step.

Ideally, at t = T-1, these two curves should be very close.
"""
function plot_value_vs_utility(world, solver_params, future_policies, utility_struct; t_check=nothing)
    (; W_grid, p_income, O_t_real_path, asset_names) = solver_params

    # Default to the last decision step if not specified
    if isnothing(t_check)
        t_check = world.config.M  # The last step before T
    end

    println("Generating Value vs Utility plot for t=$t_check...")

    # 1. Calculate Theoretical Utility U(W)
    u_vals = [utility_struct.u(w) for w in W_grid]

    # 2. Calculate "Predicted" Value Function J(W)
    # We do this by running the "calculate_expected_utility" helper for each grid point.
    # This effectively acts as J(W, t) = E_t[ J(W_next, t+1) ]

    # We need a dummy policy for the *current* step to launch the simulation.
    # We'll use the solver's own policy for this step.
    # Since the policy varies by simulation, we compute the *average* value across simulations
    # to get a representative J(W) curve.

    J_vals = zeros(length(W_grid))

    # Pre-fetch policies for this step
    policies_at_t = future_policies[t_check]
    sims = world.config.sims

    for (i, W) in enumerate(W_grid)
        # We need to construct the specific ω vectors for all simulations at this W
        # to pass into calculate_expected_utility
        current_weights = [policies_at_t[s](W) for s in 1:sims]

        # Calculate J(W)
        J_val, _ = calculate_expected_utility(
            world, solver_params, future_policies, t_check, W, current_weights, utility_struct
        )
        J_vals[i] = J_val
    end

    # 3. Plot
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1],
        title = "Value Function vs Utility (t=$t_check)",
        xlabel = "Wealth (W)",
        ylabel = "Value / Utility"
    )

    lines!(ax, W_grid, u_vals, color=:black, linestyle=:dash, linewidth=2, label="Theoretical U(W)")
    scatterlines!(ax, W_grid, J_vals, color=:red, linewidth=2, label="Predicted J(W)")

    axislegend(ax, position=:rb)

    return fig
end

# ==============================================================================
# 5. DIAGNOSTIC: POLICY SURFACE (Wealth x State)
# ==============================================================================
"""
    plot_policy_surface(future_policies, world, solver_params, t_idx, state_name::Symbol)

Creates a 3D Surface plot showing how the Optimal Weight varies with both Wealth (X-axis)
and a State Variable (Y-axis).
"""
function plot_policy_surface(future_policies, world, solver_params, t_idx, state_name::Symbol)
    (; asset_names, W_grid) = solver_params

    # 1. Get State Variable Data (Y-axis)
    if !hasproperty(world.paths, state_name)
        error("State variable :$state_name not found.")
    end
    state_col = getproperty(world.paths, state_name)[:, t_idx]

    # We need a sorted grid for the surface plot
    # We'll sample 50 points from the state distribution to form a regular grid
    y_min, y_max = minimum(state_col), maximum(state_col)
    state_grid = range(y_min, y_max, length=50)

    # 2. Prepare Data for Surface
    # Z-matrix dimensions: (length(W_grid), length(state_grid))
    n_assets = length(asset_names)

    # We create a figure with one surface per asset
    fig = Figure(size = (800, 600 * n_assets))

    # Extract policies
    policies_at_t = future_policies[t_idx]

    for (k, asset) in enumerate(asset_names)
        Z_matrix = zeros(length(W_grid), length(state_grid))

        # Fill the matrix
        # For every point (W, State), we need to find the policy.
        # Since our policies are indexed by simulation (i.e., by State),
        # we need to find the "nearest neighbor" simulation for each state grid point.

        sorted_indices = sortperm(state_col)
        sorted_states = state_col[sorted_indices]

        for (j, s_val) in enumerate(state_grid)
            # Find closest simulation index
            idx = searchsortedfirst(sorted_states, s_val)
            idx = clamp(idx, 1, length(sorted_states))
            sim_idx = sorted_indices[idx]

            # Evaluate that simulation's policy for all W
            for (i, w_val) in enumerate(W_grid)
                weight = policies_at_t[sim_idx](w_val)[k]
                Z_matrix[i, j] = weight
            end
        end

        # 3. Plot Surface
        ax = Axis3(fig[k, 1],
            title = "Optimal Allocation to $asset (t=$t_idx)",
            xlabel = "Wealth (W)",
            ylabel = "$state_name",
            zlabel = "Weight (ω)"
        )

        surface!(ax, W_grid, collect(state_grid), Z_matrix, colormap=:viridis)
    end

    return fig
end