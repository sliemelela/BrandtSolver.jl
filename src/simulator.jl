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