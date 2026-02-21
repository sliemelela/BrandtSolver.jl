"""
    simulate_wealth_trajectory(
        W_start_vec::Vector{Float64},
        t_start_idx::Int,
        T_end_idx::Int,
        X_full::AbstractMatrix{Float64},
        Y_full::AbstractMatrix{Float64},
        Re_full::AbstractMatrix{<:SVector},
        future_policies::Vector{Vector{Any}};
        forced_policy_at_t_start=nothing,
        recorder::AbstractPathRecorder = NoOpPathRecorder()
    )

Simulates wealth trajectories forward in time across all simulation paths from `t_start_idx` to
`T_end_idx`.

At each time step, the function determines the portfolio weights
(either using `forced_policy_at_t_start` for the initial step or querying the interpolated
`future_policies` for subsequent steps).
It then advances wealth using the `calculate_next_wealth` physics kernel.

# Arguments
- `W_start_vec::Vector{Float64}`: Initial wealth values for all simulation paths at `t_start_idx`.
- `t_start_idx::Int`: The starting time step index.
- `T_end_idx::Int`: The terminal time step index.
- `X_full::AbstractMatrix{Float64}`: Abstract matrix `(sims × steps)` of gross risk-free returns.
- `Y_full::AbstractMatrix{Float64}`: Abstract matrix `(sims × steps)` of non-tradable income yields.
- `Re_full::AbstractMatrix{<:SVector}`: Abstract matrix `(sims × steps)` of `SVector` excess returns.
- `future_policies::Vector{Vector{Any}}`: A nested vector of interpolated policies for future time steps.
- `forced_policy_at_t_start`: Optional portfolio weight(s) to strictly apply at `t_start_idx`.
    Can be a single `SVector` or a `Vector` of `SVector`s.
- `recorder::AbstractPathRecorder`: A logging mechanism to record the simulated wealth paths (defaults to `NoOpPathRecorder()`).

# Returns
- A tuple `(W_t_plus_1, W_T)`:
  - `W_t_plus_1::Vector{Float64}`: Wealth at the immediate next time step (`t_start_idx + 1`).
    This is captured and returned because it acts as the base expansion point in the denominator
    of the Brandt equation.
  - `W_T::Vector{Float64}`: Terminal wealth at `T_end_idx`.
"""
function simulate_wealth_trajectory(
    W_start_vec::Vector{Float64},
    t_start_idx::Int,
    T_end_idx::Int,
    X_full::AbstractMatrix{Float64},
    Y_full::AbstractMatrix{Float64},
    Re_full::AbstractMatrix{<:SVector},
    future_policies::Vector{Vector{Any}};
    forced_policy_at_t_start=nothing,
    recorder::AbstractPathRecorder = NoOpPathRecorder()
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
    calculate_realized_term(
        n::Int,
        k_vec::Vector{Int},
        W_t_plus_1::Vector{Float64},
        W_T::Vector{Float64},
        Re_next_svec::AbstractVector{<:SVector},
        utility::UtilityFunctions,
    )

Computes the realized marginal utility integrand (``Y_n``) for the ``n``-th order term of the
Taylor-expanded equation across all simulation paths.

The realized integrand for the Taylor expansion evaluates to:
```math
    Y_n = U^{(n)}(W_T) \\left(\\frac{W_T}{W_{t+1}}\\right)^n \\left(\\prod_{j=1}^{N} R_{e, j, t+1}^{k_j}\\right) R_{e, t+1}
````

# Arguments
- `n::Int`: The specific derivative order in the Taylor expansion (e.g., 2 for the variance term).
- `k_vec::Vector{Int}`: A vector of integers representing the cross-asset monomial exponents \$k_j\$ for the multivariate expansion.
- `W_t_plus_1::Vector{Float64}`: Simulated wealth at time \$t+1\$.
- `W_T::Vector{Float64}`: Simulated terminal wealth at time \$T\$.
- `Re_next_svec::AbstractVector{<:SVector}`: A vector of `SVector` excess returns at time \$t+1\$.
- `utility::UtilityFunctions`: The `UtilityFunctions` struct containing the \$n\$-th derivative generator.

# Returns
- A `Matrix{Float64}` of size `(sims × N_assets)` containing the evaluated integrand for each simulation path and each asset.
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
    Y_n = Matrix{Float64}(undef, sim, N_assets)

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