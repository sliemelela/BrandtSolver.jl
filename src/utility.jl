"""
    create_utility_from_ad(base_utility_func::Function)

Wraps a user-provided base utility function into a [`UtilityFunctions`](@ref) struct,
automatically generating its exact higher-order derivatives and an inverse function.

# Details
- **Derivatives:** Uses `ForwardDiff.derivative` recursively to compute any ``n``-th order derivative.
    Results are aggressively cached in a closure dictionary so that re-evaluating the same
    derivative order carries zero compilation or setup overhead.
- **Inverse:** Generates an inverse utility function ``W = U^{-1}(J)`` using Newton's method
    (`Roots.find_zero`), relying on the automatically computed first derivative for fast convergence.

# Arguments
- `base_utility_func::Function`: A standard scalar function representing utility,
    e.g., ``W \\mapsto \\frac{W^{1-γ}}{1-γ}``

# Returns
- A populated `UtilityFunctions` struct.
"""
function create_utility_from_ad(base_utility_func::Function)

    # 1. Initialize the Cache
    # This Dict lives inside the closure and persists as long as the struct exists.
    # Map: Order (Int) -> Derivative Function
    deriv_cache = Dict{Int, Function}()

    # Base case: 0-th derivative is the function itself
    deriv_cache[0] = base_utility_func

    # 2. Define the Recursive Caching Function
    function get_deriv(n::Int)
        # Check cache first
        if haskey(deriv_cache, n)
            return deriv_cache[n]
        end

        # If not found, get the (n-1)th derivative (recursive call)
        # This naturally builds up the chain: 0 -> 1 -> 2 -> ... -> n
        prev_deriv = get_deriv(n - 1)

        # Create the new derivative using ForwardDiff
        # We wrap it in a generic function x -> ...
        new_deriv = x -> ForwardDiff.derivative(prev_deriv, x)

        # Store in cache
        deriv_cache[n] = new_deriv

        return new_deriv
    end

    # Create Inverse Utility (Same as before)
    function inverse_utility(target_val)
        f(W) = base_utility_func(W) - target_val
        W_initial_guess = 1.0
        # Uses the cached derivative for the first derivative of the base utility
        W_solution = find_zero((f, deriv_cache[1]), W_initial_guess, Roots.Newton())
        return W_solution
    end

    # Return the updated struct
    return UtilityFunctions(
        u = base_utility_func,
        nth_derivative = get_deriv, # Pass the getter function
        inverse = inverse_utility
    )
end

"""
     calculate_next_wealth(W_current, ω_t, Re_next, R_free)

The core physics kernel that advances wealth...

# Arguments
- `W_current`: The wealth at time ``t``.
- `ω_t`: The chosen portfolio weight(s) for the risky assets at time ``t``.
- `Re_next`: The realized excess return(s) of the risky assets at time ``t+1``.
- `R_free`: The gross risk-free return between ``t`` and ``t+1``.
"""
function calculate_next_wealth(W_current, ω_t, Re_next, R_free)

    # Portfolio Return Component
    portfolio_return = dot.(ω_t, Re_next)

    # Total Growth Factor
    growth_factor = portfolio_return .+ R_free

    # Update Wealth
    return W_current .* growth_factor
end
