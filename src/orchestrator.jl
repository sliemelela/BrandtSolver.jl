"""

    compute_expectations_and_policy(
        t_decision::Int,
        poly_order::Int,
        max_taylor_order::Int,
        W_at_t::Float64,
        Z_all_paths::Vector{<:AbstractMatrix{Float64}},
        R_free_base::AbstractMatrix{Float64},
        income_component::AbstractMatrix{Float64},
        Re_all_paths::AbstractMatrix{<:SVector},
        T_steps::Int,
        utility::UtilityFunctions,
        future_policies::Vector,
        reg_strategy::RegressionStrategy,
    )

Calculates the conditional expectations of utility moments and solves the First Order Conditions
(FOCs) for the optimal portfolio weights across all simulation paths at a specific decision time
and a *single* starting wealth level.

# Arguments
- `t_decision::Int`: The current time step in the backward recursion.
- `poly_order::Int`: The polynomial degree for the cross-sectional regression basis.
- `max_taylor_order::Int`: The Taylor expansion order of the value function.
- `W_at_t::Float64`: The specific wealth level being evaluated across all paths.
- `Z_all_paths::Vector{<:AbstractMatrix{Float64}}`: A vector of matrices representing the state variables.
- `R_free_base::AbstractMatrix{Float64}`: A matrix of gross risk-free returns.
- `income_component::AbstractMatrix{Float64}`: A matrix of the non-tradable income yields.
- `Re_all_paths::AbstractMatrix{<:SVector}`: A matrix of SVector excess returns for the risky assets.
- `T_steps::Int`: The total number of simulation time steps.
- `utility::UtilityFunctions`: The utility struct containing `u` and its derivatives.
- `future_policies::Vector`: The optimal policy interpolators for time periods `t > t_decision`.
- `reg_strategy::RegressionStrategy`: The OLS strategy (Standard or Trimmed).

# Returns
- `ω_t_svec`: A vector of `SVector`s containing the optimal weights for each simulation path.
- `expected_moments_all`: A dictionary mapping exponent vectors to their conditionally expected matrices.
- `W_t_plus_1`: The simulated wealth at the next time step.
- `W_T`: The simulated terminal wealth.
"""
function compute_expectations_and_policy(
    t_decision::Int,
    poly_order::Int,
    max_taylor_order::Int,
    W_at_t::Float64,
    Z_all_paths::Vector{<:AbstractMatrix{Float64}},
    R_free_base::AbstractMatrix{Float64},
    income_component::AbstractMatrix{Float64},
    Re_all_paths::AbstractMatrix{<:SVector},
    T_steps::Int,
    utility::UtilityFunctions,
    future_policies::Vector,
    reg_strategy::RegressionStrategy,
)
    sim = size(R_free_base, 1)
    N_assets = length(Re_all_paths[1,1])
    zero_policy = SVector{N_assets, Float64}(zeros(N_assets))

    W_t_plus_1, W_T = simulate_wealth_trajectory(
        fill(W_at_t, sim),
        t_decision,
        T_steps,
        R_free_base,
        income_component,
        Re_all_paths,
        future_policies;
        forced_policy_at_t_start = zero_policy,
        recorder = NoOpPathRecorder()
    )

    # Re_next_svec is a view of the matrix column
    Re_next_svec = view(Re_all_paths, :, t_decision + 1)

    expected_moments_all = compute_and_regress_moments(
        t_decision, poly_order, max_taylor_order, Z_all_paths,
        W_t_plus_1, W_T, Re_next_svec,
        utility, reg_strategy,
    )

    ω_t_svec = Vector{SVector{N_assets, Float64}}(undef, sim)

    # Optimization loop (could be threaded)
    for i in 1:sim
        moments_i = Dict{Vector{Int}, Vector{Float64}}()
        for (k, mat) in expected_moments_all
            moments_i[k] = mat[i, :]
        end

        sol = solve_higher_order_policy(moments_i, W_at_t, N_assets, max_taylor_order)
        ω_t_svec[i] = SVector{N_assets, Float64}(sol...)
    end

    return ω_t_svec, expected_moments_all, W_t_plus_1, W_T
end

"""
    create_policy_interpolators(
        t_decision::Int,
        W_grid::Vector{Float64},
        poly_order::Int,
        Z_all_paths::Vector{<:AbstractMatrix{Float64}},
        R_free_base::AbstractMatrix{Float64},
        income_component::AbstractMatrix{Float64},
        Re_all_paths::AbstractMatrix{<:SVector},
        T_steps::Int,
        utility::UtilityFunctions,
        future_policies::Vector{Vector{Any}},
        max_taylor_order::Int,
        reg_strategy::RegressionStrategy,
        recorder::AbstractSolverRecorder
    )

Constructs state-contingent policy rules for a given decision time by evaluating optimal portfolio
weights across a specified grid of wealth values (`W_grid`) and interpolating the results.

# Arguments
- `t_decision::Int`: Current time step in the backward recursion.
- `W_grid::Vector{Float64}`: The grid of wealth values to evaluate.
- `poly_order::Int`: Polynomial degree for the cross-sectional regression.
- `Z_all_paths::Vector{<:AbstractMatrix{Float64}}`: Vector of state variable matrices.
- `R_free_base::AbstractMatrix{Float64}`: Gross risk-free return matrix.
- `income_component::AbstractMatrix{Float64}`: Non-tradable income matrix.
- `Re_all_paths::AbstractMatrix{<:SVector}`: Matrix of `SVector` excess returns.
- `T_steps::Int`: Total number of time steps.
- `utility::UtilityFunctions`: The `UtilityFunctions` struct.
- `future_policies::Vector{Vector{Any}}`: A vector of previously computed future policies.
- `max_taylor_order::Int`: Expansion order for the Euler equation.
- `reg_strategy::RegressionStrategy`: The OLS strategy (Standard or Trimmed).
- `recorder::AbstractSolverRecorder`: The logging mechanism for debugging data.

# Returns
- A `Vector` of linear interpolation objects (one for each simulation path).
    Each interpolator takes a wealth value `W` and returns the optimal portfolio allocation vector `ω`.
"""
function create_policy_interpolators(
    t_decision::Int,
    W_grid::Vector{Float64},
    poly_order::Int,
    Z_all_paths::Vector{<:AbstractMatrix{Float64}},
    R_free_base::AbstractMatrix{Float64},
    income_component::AbstractMatrix{Float64},
    Re_all_paths::AbstractMatrix{<:SVector},
    T_steps::Int,
    utility::UtilityFunctions,
    future_policies::Vector{Vector{Any}},
    max_taylor_order::Int,
    reg_strategy::RegressionStrategy,
    recorder::AbstractSolverRecorder
)
    max_taylor_order > 1 || throw(DomainError(max_taylor_order, "Taylor order must be > 1."))

    sim = size(Re_all_paths, 1)
    N_assets = length(Re_all_paths[1,1])
    policy_on_grid = Matrix{SVector{N_assets, Float64}}(undef, sim, length(W_grid))

    for (i, W_at_t) in enumerate(W_grid)
        # Optional: Print progress
        # println("... calculating policy at t=$t_decision for W=$W_at_t")

        ω_t_at_W, _, W_t_plus_1, W_T = compute_expectations_and_policy(
            t_decision, poly_order, max_taylor_order, W_at_t,
            Z_all_paths, R_free_base, income_component,
            Re_all_paths, T_steps, utility, future_policies, reg_strategy,
        )

        record_step!(recorder, t_decision, i, :W_at_t, W_at_t)
        record_step!(recorder, t_decision, i, :W_at_t_plus_1, W_t_plus_1)
        record_step!(recorder, t_decision, i, :policy, ω_t_at_W)

        policy_on_grid[:, i] = ω_t_at_W
    end

    policy_interpolators = Vector{Any}(undef, sim)
    for i in 1:sim
        policy_for_sim_i_on_grid = view(policy_on_grid, i, :)
        policy_interpolators[i] = linear_interpolation(
            W_grid,
            policy_for_sim_i_on_grid,
            extrapolation_bc = Line()
        )
    end

    return policy_interpolators
end


"""
    solve_portfolio_problem(
        Re_all_paths::AbstractMatrix{<:SVector},
        Z_all_paths::Vector{<:AbstractMatrix{Float64}},
        X_all_paths::AbstractMatrix{Float64},
        Y_all_paths::AbstractMatrix{Float64},
        solver_params::SolverParams,
        utility::UtilityFunctions;
        recorder::AbstractSolverRecorder = NoOpRecorder()
    )
The primary entry point for solving the dynamic portfolio choice problem.

It orchestrates the backward recursion (dynamic programming) by iterating backwards from the period
before terminal time, `T-1`, to period `1`.
At each step, it calculates the optimal policies conditionally based on the optimal `future_policies`
derived in the previous iterations.

# Arguments
- `Re_all_paths::AbstractMatrix{<:SVector}`: A matrix `(sims × steps)` of `SVector`s representing excess returns.
- `Z_all_paths::Vector{<:AbstractMatrix{Float64}}`: A vector containing the state variable matrices.
- `X_all_paths::AbstractMatrix{Float64}`: The base gross risk-free return matrix.
- `Y_all_paths::AbstractMatrix{Float64}`: The non-tradable income return matrix.
- `solver_params::SolverParams`: Configuration parameters (e.g., `W_grid`, polynomial orders).
- `utility::UtilityFunctions`: The configured utility and derivatives functions.
- `recorder::AbstractSolverRecorder`: An optional recorder to log intermediate steps (defaults to `NoOpRecorder()`).

# Returns
- `future_policies`: A nested vector structure `[time_step][simulation_path](Wealth)` that holds
    the full interpolated policy rules for every path at every time step.
"""
function solve_portfolio_problem(
    Re_all_paths::AbstractMatrix{<:SVector},
    Z_all_paths::Vector{<:AbstractMatrix{Float64}},
    X_all_paths::AbstractMatrix{Float64},
    Y_all_paths::AbstractMatrix{Float64},
    solver_params::SolverParams,
    utility::UtilityFunctions;
    recorder::AbstractSolverRecorder = NoOpRecorder()
)
    (; W_grid, poly_order, max_taylor_order, trimming_α) = solver_params

    sim, T_steps = size(X_all_paths)

    reg_strategy = trimming_α > 0.0 ? TrimmedOLS(trimming_α) : StandardOLS()

    future_policies = Vector{Vector{Any}}(undef, T_steps)

    for t_decision in (T_steps - 1):-1:1
        println("--- Processing Policy for Decision Time t = $t_decision ---")

        future_policies[t_decision] = create_policy_interpolators(
            t_decision, W_grid, poly_order, Z_all_paths, X_all_paths,
            Y_all_paths, Re_all_paths, T_steps, utility, future_policies,
            max_taylor_order, reg_strategy, recorder
        )
    end

    println("--- Backwards Recursion Complete ---")
    return future_policies
end

"""
    calculate_expected_utility(
        Re_all_paths::AbstractMatrix{<:SVector},
        X_all_paths::AbstractMatrix{Float64},
        Y_all_paths::AbstractMatrix{Float64},
        future_policies::Vector{Vector{Any}},
        t_start::Int,
        W_start::Float64,
        ω_force::Union{Vector{<:SVector}, <:SVector},
        utility_struct::UtilityFunctions
    )

Evaluates the expected utility and Certainty Equivalent (CE) of a specific portfolio allocation
(`ω_force`) at time `t_start` for a given starting wealth (`W_start`).

It calculates this by forward-simulating the wealth trajectories from `t_start` to terminal time `T`,
assuming the agent applies the forced weights at `t_start` and then follows the optimal
`future_policies` for all subsequent periods.

Paths that result in bankruptcy (wealth ``\\leq 10^{-9}``) are filtered out to prevent
numerical explosion (e.g., utility approaching ``-\\infty`` for CRRA) before computing the final mean.

# Arguments
- `Re_all_paths::AbstractMatrix{<:SVector}`: A matrix `(sims × steps)` of `SVector`s representing excess returns.
- `X_all_paths::AbstractMatrix{Float64}`: A matrix `(sims × steps)` of gross risk-free returns.
- `Y_all_paths::AbstractMatrix{Float64}`: A matrix `(sims × steps)` of non-tradable income yields.
- `future_policies::Vector{Vector{Any}}`: A nested vector structure containing the
    interpolated policy rules for ``t > t_{start}``.
- `t_start::Int`: The time step from which to begin the forward simulation.
- `W_start::Float64`: The starting wealth level applied uniformly across all simulation paths.
- `ω_force::Union{Vector{<:SVector}, <:SVector}`: The portfolio allocation to force at `t_start`.
    Can be a single `SVector` (applied to all paths) or a `Vector` of `SVector`s (path-dependent weights).
- `utility_struct::UtilityFunctions`: The configured `UtilityFunctions` containing the base
    utility `u` and its `inverse`.

# Returns
- A tuple `(J_0, CE_0)`:
  - `J_0::Float64`: The conditionally expected utility across all valid, non-bankrupt simulated paths.
  - `CE_0::Float64`: The Certainty Equivalent of that expected utility (the guaranteed, risk-free wealth level that yields identical utility).
"""
function calculate_expected_utility(
    Re_all_paths::AbstractMatrix{<:SVector},
    X_all_paths::AbstractMatrix{Float64},
    Y_all_paths::AbstractMatrix{Float64},
    future_policies::Vector{Vector{Any}},
    t_start::Int,
    W_start::Float64,
    ω_force::Union{Vector{<:SVector}, <:SVector},
    utility_struct::UtilityFunctions
)
    sim, T_steps = size(X_all_paths)

    _, W_T = simulate_wealth_trajectory(
        fill(W_start, sim),
        t_start,
        T_steps,
        X_all_paths,
        Y_all_paths,
        Re_all_paths,
        future_policies;
        forced_policy_at_t_start = ω_force,
        recorder = NoOpPathRecorder()
    )

    W_valid = filter(w -> w > 1e-9, W_T)
    if isempty(W_valid)
        return (-Inf, 0.0)
    end

    J_0 = mean(utility_struct.u.(W_valid))
    CE_0 = utility_struct.inverse(J_0)

    return J_0, CE_0
end