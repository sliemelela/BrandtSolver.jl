using LinearAlgebra
using Statistics
using StaticArrays
using ForwardDiff
using Interpolations
using Combinatorics
using NonlinearSolve
using FinancialMarketSimulation

# ==============================================================================
# 1. CORE SIMULATION & MATH
# ==============================================================================

"""
    simulate_wealth_trajectory(...)

Universal simulator.
Fixed: Now accepts AbstractMatrix for X, Y, Re to handle ComponentArray views.
"""
function simulate_wealth_trajectory(
    W_start_vec::Vector{Float64},
    t_start_idx::Int,
    T_end_idx::Int,
    X_full::AbstractMatrix{Float64},      # <--- CHANGED
    Y_full::AbstractMatrix{Float64},      # <--- CHANGED
    Re_full::AbstractMatrix{<:SVector},   # <--- CHANGED
    future_policies::Vector{Vector{Any}};
    forced_policy_at_t_start=nothing,
    recorder::AbstractSolverRecorder = NoOpRecorder()
)
    sim = length(W_start_vec)
    W_current = copy(W_start_vec)

    # Record initial state if supported
    if hasmethod(record_wealth!, Tuple{typeof(recorder), Int, Vector{Float64}})
        record_wealth!(recorder, 1, W_current)
    end

    W_t_plus_1 = Vector{Float64}(undef, sim)

    for (k, t) in enumerate(t_start_idx:(T_end_idx - 1))

        # Policy Logic
        if k == 1 && forced_policy_at_t_start !== nothing
            if forced_policy_at_t_start isa SVector
                ω_t = [forced_policy_at_t_start for _ in 1:sim]
            else
                ω_t = forced_policy_at_t_start
            end
        else
            policies_at_t = future_policies[t]
            ω_t = [policies_at_t[i](W_current[i]) for i in 1:sim]
        end

        # Market Evolution
        # X[:, t] and Y[:, t] might be views, which is fine for broadcasting
        R_free = X_full[:, t] .+ Y_full[:, t] ./ W_current
        Re_next = Re_full[:, t + 1]

        # Update Wealth (Math function from utility.jl)
        W_current = calculate_next_wealth(W_current, ω_t, Re_next, R_free)

        # Snapshot for regression
        if k == 1
            W_t_plus_1 .= W_current
        end

        if hasmethod(record_wealth!, Tuple{typeof(recorder), Int, Vector{Float64}})
            record_wealth!(recorder, k + 1, W_current)
        end
    end

    return (W_t_plus_1, W_current)
end

"""
    calculate_realized_term(...)

Calculates Y_n integrand.
"""
function calculate_realized_term(
    n::Int,
    k_vec::Vector{Int},
    W_t_plus_1::Vector{Float64},
    W_T::Vector{Float64},
    Re_next_svec::AbstractVector{<:SVector},
    utility::UtilityFunctions,
)
    sim = length(W_T)
    N_assets = length(Re_next_svec[1])
    Y_n = Matrix{Float64}(undef, sim, N_assets) # Output is always dense matrix

    ∂u_func = utility.nth_derivative(n)

    for s in 1:sim
        ∂u_val = ∂u_func(W_T[s])
        wealth_adj = (W_T[s] / W_t_plus_1[s])^n
        Re_vec = Re_next_svec[s]
        monomial_val = prod(Re_vec .^ k_vec)

        Y_n[s, :] .= (∂u_val * wealth_adj * monomial_val) .* Re_vec
    end

    return Y_n
end

# ==============================================================================
# 2. REGRESSION & EXPECTATIONS
# ==============================================================================

function compute_and_regress_moments(
    t_decision::Int,
    poly_order::Int,
    max_taylor_order::Int,
    Z_all_paths::Vector{<:AbstractMatrix{Float64}}, # <--- CHANGED
    W_t_plus_1::Vector{Float64},
    W_T::Vector{Float64},
    Re_next_svec::AbstractVector{<:SVector},
    utility::UtilityFunctions,
    reg_strategy::RegressionStrategy,
)
    N_assets = length(Re_next_svec[1])
    expected_moments = Dict{Vector{Int}, Matrix{Float64}}()

    # Pre-compute Regression Context
    φ = nothing
    reg_context = nothing

    if t_decision > 1
        # Extract columns (might be views, convert to vector/matrix for regression if needed)
        Z_at_t = [Z_path[:, t_decision] for Z_path in Z_all_paths]

        # Power matrix usually expects Vector{Vector} or Matrix.
        # We assume power_matrix handles Vector{AbstractVector} correctly.
        φ = power_matrix(Z_at_t, poly_order)
        reg_context = prepare_regression_context(reg_strategy, φ)
    end

    for n in 1:max_taylor_order
        for k_vec in multiexponents(N_assets, n - 1)
            Y_k = calculate_realized_term(n, k_vec, W_t_plus_1, W_T, Re_next_svec, utility)
            E_k = compute_conditional_expectation(t_decision, reg_strategy, reg_context, φ, Y_k)
            expected_moments[k_vec] = E_k
        end
    end

    return expected_moments
end

function compute_conditional_expectation(
    t_decision::Int,
    reg_strategy::RegressionStrategy,
    reg_context::Any, # Factorization or Matrix
    φ::Union{Nothing, AbstractMatrix{Float64}},
    integrand::Matrix{Float64},
)
    sim = size(integrand, 1)
    N_assets = size(integrand, 2)

    if t_decision == 1
        E_mean = mean(integrand, dims=1)
        return repeat(E_mean, sim, 1)
    end

    E_t = Matrix{Float64}(undef, sim, N_assets)
    for (j, asset_col) in enumerate(eachcol(integrand))
        θ = estimate_coefficients(reg_strategy, reg_context, asset_col)
        E_t[:, j] = φ * θ
    end

    return E_t
end

# ==============================================================================
# 3. OPTIMIZATION (FOC SOLVERS)
# ==============================================================================

function solve_2nd_order_policy(
    expected_moments_i::Dict{Vector{Int}, Vector{Float64}},
    W_t::Float64,
    N_assets::Int,
)
    k_zero = zeros(Int, N_assets)
    a_vec = expected_moments_i[k_zero]

    B_mat = Matrix{Float64}(undef, N_assets, N_assets)
    for j in 1:N_assets
        k_col = zeros(Int, N_assets)
        k_col[j] = 1
        B_mat[:, j] = expected_moments_i[k_col]
    end

    mean_diag = mean(abs.(diag(B_mat)))
    scale = mean_diag > 1e-20 ? mean_diag : 1.0
    λ = 1.0e-6 * scale
    jit = λ * I

    return -(1.0 / W_t) * (B_mat - jit) \ a_vec
end

function solve_higher_order_policy(
    expected_moments_i::Dict{Vector{Int}, Vector{Float64}},
    W_t::Float64,
    N_assets::Int,
    max_taylor_order::Int,
)
    analytical_sol = solve_2nd_order_policy(expected_moments_i, W_t, N_assets)
    if max_taylor_order == 2
        return analytical_sol
    end

    initial_guess = analytical_sol
    p = (
        expected_moments_i = expected_moments_i,
        W_t = W_t,
        N_assets = N_assets,
        max_taylor_order = max_taylor_order
    )

    function foc_system!(F, ω, p)
        expected_moments_i = p.expected_moments_i
        W_t = p.W_t
        N_assets = p.N_assets
        max_taylor_order = p.max_taylor_order

        fill!(F, 0.0)

        for n in 1:max_taylor_order
            scale_factor = (W_t^(n - 1)) / factorial(n - 1)
            for k_vec in multiexponents(N_assets, n - 1)
                coeff = multinomial(k_vec...)
                omega_monomial = prod(ω .^ k_vec)
                E_vec = expected_moments_i[k_vec]
                F .+= scale_factor * coeff * omega_monomial .* E_vec
            end
        end
    end

    prob = NonlinearProblem(foc_system!, initial_guess, p)
    sol = solve(prob)

    if SciMLBase.successful_retcode(sol)
        return sol.u
    else
        return initial_guess
    end
end

# ==============================================================================
# 4. ORCHESTRATION
# ==============================================================================

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
        recorder = NoOpRecorder()
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
    world::SimulationWorld,
    solver_params::SolverParams,
    utility::UtilityFunctions;
    O_t_real_path::Union{String, Nothing} = nothing,
    p_income::Float64 = 0.0,
    recorder::AbstractSolverRecorder = NoOpRecorder()
)
    (; asset_names, state_names, W_grid, poly_order, max_taylor_order,
        trimming_α) = solver_params

    # API Compatibility
    sim = world.config.sims
    M   = world.config.M
    T_steps = M + 1

    reg_strategy = trimming_α > 0.0 ? TrimmedOLS(trimming_α) : StandardOLS()

    # Data Packaging
    Re_all_paths = package_excess_returns(world, asset_names)
    Z_all_paths = get_state_variables(world, state_names)

    if O_t_real_path === nothing
        O_t_real = zeros(sim, T_steps)
    else
        O_t_real = getproperty(world.paths, Symbol(O_t_real_path))
    end

    R_free_base, income_component = create_risk_free_return_components(
        world, p_income, O_t_real
    )

    future_policies = Vector{Vector{Any}}(undef, T_steps)

    for t_decision in (T_steps - 1):-1:1
        println("--- Processing Policy for Decision Time t = $t_decision ---")

        future_policies[t_decision] = create_policy_interpolators(
            t_decision, W_grid, poly_order, Z_all_paths, R_free_base,
            income_component, Re_all_paths, T_steps, utility, future_policies,
            max_taylor_order, reg_strategy, recorder
        )
    end

    println("--- Backwards Recursion Complete ---")
    return future_policies
end

function calculate_expected_utility(
    world,
    solver_params,
    future_policies,
    t_start,
    W_start,
    ω_force,
    utility_struct;
    p_income::Float64 = 0.0,
    O_t_real_path::Union{String, Nothing} = nothing,
)
    sim = world.config.sims
    T_steps = world.config.M + 1

    if O_t_real_path === nothing
        O_t_full = zeros(sim, T_steps)
    else
        O_t_full = getproperty(world.paths, Symbol(O_t_real_path))
    end

    X_full, Y_full = create_risk_free_return_components(world, p_income, O_t_full)
    Re_full = package_excess_returns(world, solver_params.asset_names)

    _, W_T = simulate_wealth_trajectory(
        fill(W_start, sim),
        t_start,
        T_steps,
        X_full,
        Y_full,
        Re_full,
        future_policies;
        forced_policy_at_t_start = ω_force,
        recorder = NoOpRecorder()
    )

    W_valid = filter(w -> w > 1e-9, W_T)
    if isempty(W_valid)
        return (-Inf, 0.0)
    end

    J_0 = mean(utility_struct.u.(W_valid))
    CE_0 = utility_struct.inverse(J_0)

    return J_0, CE_0
end