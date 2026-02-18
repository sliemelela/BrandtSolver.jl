"""
    package_excess_returns(
        world::SimulationWorld,
        asset_names::Vector{Symbol}
    )

Takes individual excess return matrices from the world and packages them
into a single `Matrix{SVector}` for the high-performance solver.

This function is "general": it creates an `SVector{N, Float64}` where `N`
is the number of asset names you provide.

# Arguments
- `world::SimulationWorld`: The simulated data.
- `asset_names::Vector{Symbol}`: A vector of the *exact* keys in `world.paths`
  that hold the excess returns, in the order you want them.
  Example: `[:Re_NominalBond, :Re_InflBond, :Re_Stock]`

# Returns
- `Matrix{SVector{N, Float64}}`: A `(sim, M+1)` matrix where each element
  is an `SVector` of the `N` asset returns for that sim/time.
"""
function package_excess_returns(
    world::SimulationWorld,
    asset_names::Vector{String}
)::SimTimeSV
    # Get dimensions and types
    sim = world.config.sims
    amount_of_time_steps = world.config.M + 1

    # N is the number of assets (e.g., 3)
    N = length(asset_names)
    SVType = SVector{N, Float64}

    # Pre-fetch all source matrices
    source_matrices = [getproperty(world.paths, Symbol(name)) for name in asset_names]

    # Pre-allocate the output matrix
    Re_packaged = Matrix{SVType}(undef, sim, amount_of_time_steps)

    # Loop and build the SVectors
    for n in 1:amount_of_time_steps, s in 1:sim
        # Create the SVector and assign it

        elements_tuple = ntuple(k -> source_matrices[k][s, n], N)
        Re_packaged[s, n] = SVType(elements_tuple)
    end

    return Re_packaged
end

"""
    create_risk_free_return_components(world, p, O_t)

Extracts risk-free rate data and constructs the solver's X and Y components.
X = 1 + (r - π)dt
Y = p * Income * dt
"""
function create_risk_free_return_components(
    world::SimulationWorld,
    p::Float64,
    O_t::Union{Matrix{Float64}, Nothing}
)
    # 1. Get Parameters
    dt = world.config.dt
    sims = world.config.sims
    steps = world.config.M + 1

    # 2. Get Rates (Assumes standard names :r and :pi exist)
    # If your simulation uses different names, we might need to pass them as args.
    r = world.paths.r

    # Handle Inflation (Optional? usually required for real returns)
    if hasproperty(world.paths, :pi)
        π = world.paths.pi
    elseif hasproperty(world.paths, :π)
        π = world.paths.π
    else
        # Default to 0 inflation if not found
        π = zeros(size(r))
    end

    # 3. Handle Income
    if isnothing(O_t)
        O_t = zeros(sims, steps)
    end

    # 4. Calculate Components
    # X_n = 1 + (r_n - π_n) * dt
    X = 1.0 .+ (r .- π) .* dt

    # Y_n = p * Income_n * dt
    Y = p .* O_t .* dt

    return X, Y
end

"""
    get_state_variables(world, state_names)

Extracts the state variables (Z) used for regression.
"""
function get_state_variables(world::SimulationWorld, state_names::Vector{String})
    # Simply map the names to the paths
    return [getproperty(world.paths, Symbol(name)) for name in state_names]
end