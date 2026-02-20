
"""
    plot_policy_rules(future_policies, solver_params, t_idx, asset_names; samples=50)

Plots the optimal portfolio allocation (weights) as a function of wealth (``W``) for a specific
decision time `t_idx`.

It plots a subset of individual simulation paths to show cross-sectional dispersion
(due to varying state variables) and overlays a thick red line representing the mean policy across all simulations.

# Arguments
- `future_policies`: The nested vector of interpolated policy functions returned by the solver.
- `solver_params`: The configuration parameters, specifically used to extract the `W_grid`.
- `t_idx`: The decision time step to evaluate.
- `asset_names`: A list of strings representing the asset names, used to generate subplots.
- `samples`: (Keyword) The number of individual simulation paths to plot as semi-transparent lines
    (defaults to 50).

# Returns
- A `Makie.Figure` object containing the generated plot.
"""
function plot_policy_rules(future_policies, solver_params, t_idx, asset_names; samples=50)
    W_grid = solver_params.W_grid

    policies_at_t = future_policies[t_idx]
    total_sims = length(policies_at_t)
    indices = rand(1:total_sims, min(samples, total_sims))

    n_assets = length(asset_names)
    fig = Figure(size = (1000, 400 * n_assets))

    for (k, asset) in enumerate(asset_names)
        ax = Axis(fig[k, 1], title = "Optimal Allocation to $asset (t=$t_idx)", xlabel = "Wealth (W)", ylabel = "Weight (ω)", titlesize = 16)

        for i in indices
            weights = [policies_at_t[i](w)[k] for w in W_grid]
            lines!(ax, W_grid, weights, color=(:blue, 0.3), linewidth=1.5)
        end

        mean_weights = zeros(length(W_grid))
        for (jw, w) in enumerate(W_grid)
            vals = [policies_at_t[i](w)[k] for i in 1:total_sims]
            mean_weights[jw] = mean(vals)
        end
        lines!(ax, W_grid, mean_weights, color=:red, linewidth=3, label="Mean Policy")
        axislegend(ax)
    end
    return fig
end

"""
    plot_state_dependence(future_policies, solver_params, t_idx, state_values::AbstractVector,
        state_name::String, asset_names; fix_W=100.0)

Plots the sensitivity of the optimal portfolio weights to a specific state variable
(e.g., interest rate or dividend yield), holding wealth strictly constant at `fix_W`.

# Arguments
- `future_policies`: The nested vector of interpolated policy functions.
- `solver_params`: The configuration parameters.
- `t_idx`: The decision time step to evaluate.
- `state_values`: A vector containing the cross-sectional values of the state variable at `t_idx`.
- `state_name`: A string label for the state variable used for the x-axis.
- `asset_names`: A list of strings representing the asset names.
- `fix_W`: (Keyword) The constant wealth level at which to evaluate the policy (defaults to 100.0).

# Returns
- A `Makie.Figure` object containing the generated plot.
"""
function plot_state_dependence(future_policies, solver_params, t_idx, state_values::AbstractVector, state_name::String, asset_names; fix_W=100.0)
    policies_at_t = future_policies[t_idx]
    sims = length(policies_at_t)
    n_assets = length(asset_names)

    weights_matrix = zeros(sims, n_assets)
    for i in 1:sims
        weights_matrix[i, :] = policies_at_t[i](fix_W)
    end

    fig = Figure(size = (1000, 400 * n_assets))
    for (k, asset) in enumerate(asset_names)
        ax = Axis(fig[k, 1], title = "Sensitivity of $asset to $state_name (at W=$fix_W, t=$t_idx)", xlabel = state_name, ylabel = "Weight (ω)", titlesize = 16)

        scatter!(ax, state_values, weights_matrix[:, k], color=(:black, 0.1), markersize=4)
        p = sortperm(state_values)
        lines!(ax, state_values[p], weights_matrix[p, k], color=:red, linewidth=2)
    end
    return fig
end

"""
     plot_realized_weights(Re_all_paths, X_all_paths, Y_all_paths, times, future_policies,
        asset_names; W_init=100.0)

Simulates wealth paths forward in time using the computed optimal `future_policies` and plots
the realized trajectories of the portfolio weights over the lifecycle.

It overlays the mean realized weight and a 90% confidence interval band.
Paths where wealth falls to zero (bankruptcy) are dynamically filtered out.

# Arguments
- `Re_all_paths`: A matrix `(sims × steps)` of `SVector`s representing excess returns.
- `X_all_paths`: A matrix `(sims × steps)` of gross risk-free returns.
- `Y_all_paths`: A matrix `(sims × steps)` of non-tradable income yields.
- `times`: An iterable of time points corresponding to the decision steps.
- `future_policies`: The nested vector of interpolated policy functions.
- `asset_names`: A list of strings representing the asset names.
- `W_init`: (Keyword) The initial wealth to start the forward simulation (defaults to 100.0).

# Returns
- A `Makie.Figure` object containing the generated plot.
"""
function plot_realized_weights(Re_all_paths, X_all_paths, Y_all_paths, times, future_policies, asset_names; W_init=100.0)
    sim, T_steps = size(X_all_paths)
    n_assets = length(asset_names)

    weight_history = zeros(sim, T_steps-1, n_assets)
    W_current = fill(W_init, sim)

    for t in 1:(T_steps - 1)
        policies = future_policies[t]

        for i in 1:sim
            if isnan(W_current[i]) || W_current[i] <= 0
                weight_history[i, t, :] .= NaN
                continue
            end
            weight_history[i, t, :] = policies[i](W_current[i])
        end

        r_free = X_all_paths[:, t] .+ Y_all_paths[:, t] ./ W_current
        re_next = Re_all_paths[:, t+1]

        for i in 1:sim
            if isnan(W_current[i]); continue; end

            port_ret = dot(weight_history[i, t, :], re_next[i])
            W_next = W_current[i] * (r_free[i] + port_ret)

            if W_next <= 1e-9 || isnan(W_next)
                W_current[i] = NaN
            else
                W_current[i] = W_next
            end
        end
    end

    fig = Figure(size = (1000, 400 * n_assets))
    for (k, asset) in enumerate(asset_names)
        ax = Axis(fig[k, 1], title = "Realized Weight Path: $asset", xlabel = "Time", ylabel = "Weight", titlesize = 16)

        indices = rand(1:sim, min(50, sim))
        for i in indices
            lines!(ax, times, weight_history[i, :, k], color=(:blue, 0.15))
        end

        means, lows, highs = Float64[], Float64[], Float64[]
        for t in 1:length(times)
            valid_data = filter(!isnan, weight_history[:, t, k])
            if isempty(valid_data)
                push!(means, NaN); push!(lows, NaN); push!(highs, NaN)
            else
                push!(means, mean(valid_data)); push!(lows, quantile(valid_data, 0.05)); push!(highs, quantile(valid_data, 0.95))
            end
        end

        lines!(ax, times, means, color=:black, linewidth=2, label="Mean Path")
        band!(ax, times, lows, highs, color=(:black, 0.1), label="90% CI")
        axislegend(ax)
    end
    return fig
end

"""
    plot_value_vs_utility(Re_all_paths, X_all_paths, Y_all_paths, solver_params, future_policies,
        utility_struct; t_check=nothing)

Serves as an economic validity check by plotting the forward-simulated expected utility
(the Value function, ``J(W)``) against the theoretical terminal utility function ``U(W)``.

If the solver is working correctly, the expected utility ``J(W)`` evaluated at ``T - 1`` should be
equal to ``U(W)``, representing the value added by optimal dynamic trading.

# Arguments
- `Re_all_paths`: A matrix of `SVector`s representing excess returns.
- `X_all_paths`: A matrix of gross risk-free returns.
- `Y_all_paths`: A matrix of non-tradable income yields.
- `solver_params`: The configuration parameters used to extract `W_grid`.
- `future_policies`: The nested vector of interpolated policy functions.
- `utility_struct`: The utility container.
- `t_check`: (Keyword) The specific time step to evaluate. Defaults to `T-1`.

# Returns
- A `Makie.Figure` object containing the generated plot.
"""
function plot_value_vs_utility(Re_all_paths, X_all_paths, Y_all_paths, solver_params, future_policies, utility_struct; t_check=nothing)
    W_grid = solver_params.W_grid
    if isnothing(t_check)
        t_check = size(X_all_paths, 2) - 1
    end

    u_vals = [utility_struct.u(w) for w in W_grid]
    J_vals = zeros(length(W_grid))

    policies_at_t = future_policies[t_check]
    sims = size(X_all_paths, 1)

    for (i, W) in enumerate(W_grid)
        current_weights = [policies_at_t[s](W) for s in 1:sims]
        J_val, _ = calculate_expected_utility(
            Re_all_paths, X_all_paths, Y_all_paths, future_policies, t_check, W, current_weights, utility_struct
        )
        J_vals[i] = J_val
    end

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = "Value Function vs Utility (t=$t_check)", xlabel = "Wealth (W)", ylabel = "Value / Utility")

    lines!(ax, W_grid, u_vals, color=:black, linestyle=:dash, linewidth=2, label="Theoretical U(W)")
    scatterlines!(ax, W_grid, J_vals, color=:red, linewidth=2, label="Predicted J(W)")
    axislegend(ax, position=:rb)

    return fig
end


"""
    plot_policy_surface(future_policies, solver_params, t_idx, state_values::AbstractVector, state_name::String, asset_names)

Generates a 3D surface plot visualizing the optimal portfolio weight as a joint function of
both Wealth (``W``) and a specific state variable at decision time `t_idx`.

This provides a comprehensive view of the agent's full policy rule, showing how
intertemporal hedging demands (driven by the state variable) interact with risk aversion (driven by wealth).

# Arguments
- `future_policies`: The nested vector of interpolated policy functions.
- `solver_params`: The configuration parameters used to extract `W_grid`.
- `t_idx`: The decision time step to evaluate.
- `state_values`: A vector containing the cross-sectional values of the state variable at `t_idx`.
- `state_name`: A string label for the state variable used for the y-axis.
- `asset_names`: A list of strings representing the asset names.

# Returns
- A `Makie.Figure` object containing the generated 3D surface plot.
"""
function plot_policy_surface(future_policies, solver_params, t_idx, state_values::AbstractVector, state_name::String, asset_names)
    W_grid = solver_params.W_grid

    y_min, y_max = minimum(state_values), maximum(state_values)
    state_grid = range(y_min, y_max, length=50)

    n_assets = length(asset_names)
    fig = Figure(size = (800, 600 * n_assets))
    policies_at_t = future_policies[t_idx]

    for (k, asset) in enumerate(asset_names)
        Z_matrix = zeros(length(W_grid), length(state_grid))

        sorted_indices = sortperm(state_values)
        sorted_states = state_values[sorted_indices]

        for (j, s_val) in enumerate(state_grid)
            idx = searchsortedfirst(sorted_states, s_val)
            idx = clamp(idx, 1, length(sorted_states))
            sim_idx = sorted_indices[idx]

            for (i, w_val) in enumerate(W_grid)
                Z_matrix[i, j] = policies_at_t[sim_idx](w_val)[k]
            end
        end

        ax = Axis3(fig[k, 1], title = "Optimal Allocation to $asset (t=$t_idx)", xlabel = "Wealth (W)", ylabel = state_name, zlabel = "Weight (ω)")
        surface!(ax, W_grid, collect(state_grid), Z_matrix, colormap=:viridis)
    end
    return fig
end