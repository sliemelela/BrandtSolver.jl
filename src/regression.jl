"""
    power_matrix(Z::Vector{Vector{Float64}}, p::Int)

Constructs a design matrix (feature matrix) for a polynomial regression model.

It generates power terms (e.g., x, x^2, ..., x^p) for each predictor variable in `Z`
and prepends a column of ones to serve as the intercept term.

The final column order is:
`[intercept, Z[1]^1, Z[1]^2, ..., Z[1]^p, Z[2]^1, Z[2]^2, ..., Z[2]^p, ...]`

# Arguments
- `Z::Vector{Vector{Float64}}`: A vector of predictor variables. `n = length(Z)` is the
  number of distinct predictor variables, and `sim = length(Z[1])` is the number
  of observations (rows). All inner vectors must have the same length.
- `p::Int`: The maximum polynomial degree to compute for each predictor.

# Returns
- `Matrix{Float64}`: A design matrix of size `sim × (n*p + 1)`.

# Examples
```jldoctest
julia> X1 = [1.0, 2.0, 3.0];
julia> X2 = [4.0, 5.0, 6.0];
julia> Z = [X1, X2]; # n=2 predictors, sim=3 observations
julia> p = 2; # Max degree

julia> power_matrix(Z, p)
3×5 Matrix{Float64}:
 1.0  1.0  1.0   4.0  16.0
 1.0  2.0  4.0   5.0  25.0
 1.0  3.0  9.0   6.0  36.0
```
"""
function power_matrix(Z::Vector{Vector{Float64}}, p::Int)
    n = length(Z)
    sim = length(Z[1])
    Xmat = hcat([Z[i] .^ k for k in 1:p, i in 1:n]...)
    return hcat(ones(sim), Xmat)
end

function prepare_regression_context(::StandardOLS, Φ::Matrix{Float64})
    return qr(Φ) # Return factorization
end

function prepare_regression_context(::TrimmedOLS, Φ::Matrix{Float64})
    return Φ     # Return raw matrix
end


"""
    estimate_coefficients(::StandardOLS, qrΦ::Factorization, Y::AbstractVector{Float64})

Estimates regression coefficients using Standard Ordinary Least Squares (OLS).

This method dispatches on the `StandardOLS` strategy type. It utilizes a pre-computed
QR factorization (`qrΦ`) for maximum performance, avoiding the need to re-factorize
the design matrix at every step.

# Arguments
- `::StandardOLS`: The strategy selector. The variable name is omitted (anonymous argument)
  because the struct contains no data needed for the calculation.
- `qrΦ::Factorization`: The QR factorization of the design matrix Φ (from `qr(Φ)`).
- `Y::AbstractVector{Float64}`: The dependent variable (response vector).

# Returns
- `Vector{Float64}`: The estimated coefficients θ minimizing ||Φθ - Y||².
"""
function estimate_coefficients(::StandardOLS, qrΦ::Factorization, Y::AbstractVector{Float64})
    return qrΦ \ Y
end

"""
    estimate_coefficients(strat::TrimmedOLS, Φ::AbstractMatrix, Y::AbstractVector{Float64})

Estimates regression coefficients using α-Trimmed OLS (Least Trimmed Squares).

This method dispatches on the `TrimmedOLS` strategy. It is designed to be robust against
outliers (e.g., extreme wealth paths in long-horizon simulations) by physically removing
the top and bottom `α` fraction of the data distribution before regressing.

# Arguments
- `strategy::TrimmedOLS`: The strategy struct containing the trimming parameter `strat.alpha`.
- `Φ::AbstractMatrix`: The **raw** design matrix.
  *Note:* Unlike Standard OLS, we cannot use a pre-computed QR factorization because
  the rows included in the regression change dynamically based on the sorting of `Y`.
- `Y::AbstractVector{Float64}`: The dependent variable (response vector).

# Algorithm
1. Calculates integer indices corresponding to the `α` and `1-α` quantiles.
2. Sorts `Y` to identify the "body" of the distribution.
3. Subsets both `Y` and `Φ` to exclude the extreme tails.
4. Solves standard OLS on the remaining subset.

# Returns
- `Vector{Float64}`: The estimated coefficients θ based on the trimmed dataset.
"""
function estimate_coefficients(strategy::TrimmedOLS, Φ::AbstractMatrix, Y::AbstractVector{Float64})

    # Obtain relevant parameters
    α = strategy.α
    sim = length(Y)

    # Determine indices to keep
    k_low  = floor(Int, α * sim) + 1
    k_high = ceil(Int, (1.0 - α) * sim)

    # Sort to find the "body" of the distribution
    p = sortperm(Y)
    keep_indices = p[k_low:k_high]

    # Solve on the subset
    return Φ[keep_indices, :] \ Y[keep_indices]
end
