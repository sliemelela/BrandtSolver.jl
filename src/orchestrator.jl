
function compute_expectations_and_policy(
    t_decision::Int,
    poly_order::Int,
    max_taylor_order::Int,
    W_at_t::Float64,
    Z_all_paths::Vector{<:AbstractMatrix{Float64}}, # <--- CHANGED
    R_free_base::AbstractMatrix{Float64},           # <--- CHANGED
    income_component::AbstractMatrix{Float64},      # <--- CHANGED
    Re_all_paths::AbstractMatrix{<:SVector},        # <--- CHANGED
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

function create_policy_interpolators(
    t_decision::Int,
    W_grid::Vector{Float64},
    poly_order::Int,
    Z_all_paths::Vector{<:AbstractMatrix{Float64}}, # <--- CHANGED
    R_free_base::AbstractMatrix{Float64},           # <--- CHANGED
    income_component::AbstractMatrix{Float64},      # <--- CHANGED
    Re_all_paths::AbstractMatrix{<:SVector},        # <--- CHANGED
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