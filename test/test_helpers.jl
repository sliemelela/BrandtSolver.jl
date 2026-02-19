using StaticArrays

# Helper to extract Returns
function package_excess_returns(world, asset_names)
    sims, steps = world.config.sims, world.config.M + 1
    N = length(asset_names)
    SVType = SVector{N, Float64}
    source_matrices = [getproperty(world.paths, Symbol(name)) for name in asset_names]
    Re_packaged = Matrix{SVType}(undef, sims, steps)
    for n in 1:steps, s in 1:sims
        val_tuple = ntuple(k -> source_matrices[k][s, n], N)
        Re_packaged[s, n] = SVType(val_tuple)
    end
    return Re_packaged
end

# Helper to extract X and Y
function create_risk_free_return_components(world, p_income, O_t)
    dt, sims, steps = world.config.dt, world.config.sims, world.config.M + 1
    r = world.paths.r
    π = hasproperty(world.paths, :pi) ? world.paths.pi : zeros(size(r))
    O_t_mat = isnothing(O_t) ? zeros(sims, steps) : O_t

    X = 1.0 .+ (r .- π) .* dt
    Y = p_income .* O_t_mat .* dt
    return X, Y
end

# Helper to extract State Variables
function get_state_variables(world, state_names)
    return [getproperty(world.paths, Symbol(name)) for name in state_names]
end