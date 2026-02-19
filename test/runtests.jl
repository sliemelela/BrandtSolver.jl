using BrandtSolver
using Test

@testset "BrandtSolver.jl" begin

    # Unit Tests
    include("unit/math_tools_test.jl")
    include("unit/physics_test.jl")

    # Integration Tests
    include("integration/interface_test.jl")
    include("integration/validity_test.jl")
end