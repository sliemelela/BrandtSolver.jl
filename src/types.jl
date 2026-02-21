# -- Aliases for dimensions
"Vector of size `(sim,)` representing values across simulations."
const SimVec      = Vector{Float64}

"Vector of size `(N,)` representing values across `N` assets."
const AssetVec    = Vector{Float64}

"Matrix of size `(sim x Time)` representing scalar values over time for each simulation."
const SimTimeMat  = Matrix{Float64}

"Matrix of size `(sim x N)` representing asset values for each simulation."
const SimAssetMat = Matrix{Float64}

"Static Vector of size `(N,)` representing `N` assets."
const AssetSV     = SVector

"Matrix of size `(Sims x Time)` where each entry is an `AssetSV`."
const SimTimeSV   = Matrix{<:AssetSV}

"""
Abstract parent type for all regression strategies.
"""
abstract type RegressionStrategy end

"""
Standard Ordinary Least Squares.
Fastest method. Uses QR factorization.
"""
struct StandardOLS <: RegressionStrategy end

"""
Trimmed OLS (robust to outliers).
Removes the top and bottom α probability mass before regressing.

# Arguments
$(TYPEDFIELDS)
"""
struct TrimmedOLS <: RegressionStrategy
    "The fraction of data to trim from both tails (e.g., 0.05 for 5% trimming)."
    α::Float64
end

"""
    $(TYPEDEF)
Configuration parameters for the Brandt portfolio solver.

# Arguments
$(TYPEDFIELDS)
"""
Base.@kwdef struct SolverParams
    "Grid of wealth values at initial time t=0 used to evaluate and interpolate the policy function."
    W_grid::Vector{Float64}
    "Order of the polynomial used for expanding state variables in the cross-sectional regression."
    poly_order::Int
    "The truncation order for the Taylor expansion of the value function."
    max_taylor_order::Int
    "The α value used if a trimmed regression strategy is applied."
    trimming_α::Float64
end

"""
    $(TYPEDEF)
A container for the utility function, its derivatives, and its inverse.

# Arguments
$(TYPEDFIELDS)

Typically, you do not construct this manually. Instead, use [`create_utility_from_ad`](@ref)
to generate it automatically from a base utility function.
"""
Base.@kwdef struct UtilityFunctions
    "The base utility function `U(W)`."
    u::Function
    "A function `f(n)` that returns a function for the `n`-th derivative of `U(W)`."
    nth_derivative::Function
    "The inverse utility function, used to calculate Certainty Equivalents."
    inverse::Function
end



# Abstraction for Recorder Types
abstract type AbstractSolverRecorder end
abstract type AbstractPathRecorder end

# The silent recorder used during normal runs so performance isn't impacted.
struct NoOpRecorder <: AbstractSolverRecorder end

# The generic "record!" function does nothing by default
function record_step!(::NoOpRecorder, args...)
    return nothing
end

# The debug recorder that actually stores data
mutable struct DebugRecorder <: AbstractSolverRecorder

    # Structure: [t_decision][w_grid_index] -> Dict(:E_a => ..., :E_b => ...)
    data::Dict{Int, Dict{Int, Dict{Symbol, Any}}}

    DebugRecorder() = new(Dict{Int, Dict{Int, Dict{Symbol, Any}}}())
end

"""
    record_step!(recorder, t, w_idx, label, value)

Hooks into the solver to save data.
"""
function record_step!(rec::DebugRecorder, t::Int, w_idx::Int, label::Symbol, value)
    # Ensure nested dict structure exists
    if !haskey(rec.data, t)
        rec.data[t] = Dict{Int, Dict{Symbol, Any}}()
    end
    if !haskey(rec.data[t], w_idx)
        rec.data[t][w_idx] = Dict{Symbol, Any}()
    end

    # Store a COPY of the value (important for arrays/mutable objects)
    rec.data[t][w_idx][label] = deepcopy(value)
end

# Helper to retrieve data easily later
function get_recorded_data(rec::DebugRecorder, t::Int, w_idx::Int, label::Symbol)
    return rec.data[t][w_idx][label]
end

# Silent Path Recorder Types
struct NoOpPathRecorder <: AbstractPathRecorder end

# The Full Recorder (For Plotting, Wraps the matrix)
struct FullPathRecorder <: AbstractPathRecorder
    path_matrix::Matrix{Float64}
end

# The generic "record!" function does nothing by default
function record_wealth!(::NoOpPathRecorder, k::Int, W_vec::Vector{Float64})
    return nothing
end

# Record wealth into the path matrix
function record_wealth!(rec::FullPathRecorder, k::Int, W_vec::Vector{Float64})
    # k is the index in the path matrix
    # W_vec is the current wealth vector
    rec.path_matrix[:, k] .= W_vec
    return nothing
end

