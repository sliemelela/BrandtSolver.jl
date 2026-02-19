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