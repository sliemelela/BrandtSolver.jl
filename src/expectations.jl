"""
    compute_and_regress_moments(t_decision, poly_order, max_taylor_order, Z_all_paths, W_t_plus_1,
        W_T, Re_next_svec, utility, reg_strategy)

Orchestrates the calculation of conditionally expected marginal utility moments for the
Taylor-expanded equation.

For each term in the Taylor expansion up to `max_taylor_order` and for all cross-asset combinations
(generated via `multiexponents`), this function calculates the realized ex-post integrand and
then projects it onto the space of available information at `t_decision` using a cross-sectional regression.

# Arguments
- `t_decision::Int`: The current time step in the backward recursion.
- `poly_order::Int`: The polynomial degree used to construct the basis functions (design matrix) from the state variables.
- `max_taylor_order::Int`: The highest order of the Taylor expansion to compute.
- `Z_all_paths::Vector{<:AbstractMatrix{Float64}}`: A vector containing the full simulated paths of all state variables.
- `W_t_plus_1::Vector{Float64}`: The simulated wealth across all paths at time ``t+1``.
- `W_T::Vector{Float64}`: The simulated terminal wealth across all paths at time ``T``.
- `Re_next_svec::AbstractVector{<:SVector}`: A vector of `SVector`s representing realized excess returns at time ``t+1``.
- `utility::UtilityFunctions`: The configured utility container with automatic differentiation.
- `reg_strategy::RegressionStrategy`: The chosen regression strategy (e.g., `StandardOLS` or `TrimmedOLS`).

# Returns
- `Dict{Vector{Int}, Matrix{Float64}}`: A dictionary mapping the cross-asset monomial exponent
    vectors (`k_vec`) to their conditionally expected values (a `sims × N_assets` matrix).
"""
function compute_and_regress_moments(
    t_decision::Int,
    poly_order::Int,
    max_taylor_order::Int,
    Z_all_paths::Vector{<:AbstractMatrix{Float64}},
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


"""
    compute_conditional_expectation(t_decision, reg_strategy, reg_context, φ, integrand)

Performs the cross-sectional regression to map realized future quantities to present expectations, conditional on current state variables.

At ``t > 1``, it uses the pre-computed regression context (e.g., a QR factorization) to rapidly
project the realized `integrand` onto the basis matrix `φ`.
At ``t = 1``, since all agents share the exact same starting state, the conditional expectation
is mathematically identical to the unconditional cross-sectional mean.

# Arguments
- `t_decision::Int`: The current time step.
- `reg_strategy::RegressionStrategy`: The specific algorithm used to estimate coefficients.
- `reg_context::Any`: The pre-computed regression context
    (e.g., a `Factorization` object for `StandardOLS` or a raw matrix for `TrimmedOLS`).
- `φ::Union{Nothing, AbstractMatrix{Float64}}`: The polynomial expansion of the state variables
    (design matrix). Will be `nothing` at ``t = 1``.
- `integrand::Matrix{Float64}`: The realized future values (``Y_n``) to be projected onto the state space.

# Returns
- `Matrix{Float64}`: The predicted (conditionally expected) values for each simulation path and each asset.
"""
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