
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