# -- Aliases for dimensions
const SimVec      = Vector{Float64}   # Size: (sim,)
const AssetVec    = Vector{Float64}   # Size: (N,), where where N is amount of assets.
const SimTimeMat  = Matrix{Float64}   # Size: (sim x Time)
const SimAssetMat = Matrix{Float64}   # Size: (sim x N), where where N is amount of assets.
const AssetSV     = SVector           # Size: (N,), where N is amount of assets.
const SimTimeSV   = Matrix{<:AssetSV} # Size: (Sims x Time) of AssetSV vectors each entry

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
"""
struct TrimmedOLS <: RegressionStrategy
    α::Float64
end

Base.@kwdef struct SolverParams
    asset_names::Vector{String}
    state_names::Vector{String}
    W_grid::Vector{Float64}
    poly_order::Int
    max_taylor_order::Int
    p_income::Float64
    O_t_real_path::Union{String, Nothing}
    trimming_α::Float64
    γ::Float64
end

"""
A container for the utility function's derivatives.
(This struct is created by `utils.jl`, not loaded)
"""
Base.@kwdef struct UtilityFunctions
    u::Function
    first_derivative::Function   # Keep for backward compatibility
    second_derivative::Function  # Keep for backward compatibility
    nth_derivative::Function     # The cached accessor function
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

