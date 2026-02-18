using BrandtSolver
using FinancialMarketSimulation
using StaticArrays
using Test

@testset "Interface Integration Tests" begin
    # ==========================================================================
    # 1. SETUP: Mocking the World
    # ==========================================================================
    # We need to create a valid SimulationWorld so that world.paths (ComponentArray)
    # has the correct keys and structure.

    # Define dummy processes to generate the necessary keys
    r_proc = VasicekProcess(:r, 0.1, 0.05, 0.01, 0.05, 1)
    pi_proc = VasicekProcess(:pi, 0.1, 0.02, 0.01, 0.02, 2)

    # We need a return path key. We define a stock and its excess return.
    stock_proc = GenericSDEProcess(:S, (t,x)->0.0, (t,x)->0.0, 100.0, [3])
    re_proc = ExcessReturnProcess(:Re_Stock, :S, :r)

    # Configure small world (sims=5, M=2 implies dt = 1.0/2 = 0.5)
    config = MarketConfig(
        sims = 5,
        T = 1.0,
        M = 2,
        processes = [r_proc, pi_proc, stock_proc, re_proc]
    )

    # Build the container
    world = build_world(config)

    # ==========================================================================
    # 2. INJECTION: Overwrite with Deterministic Data
    # ==========================================================================
    # We ignore the actual simulation results and inject our known values
    # to test the math of the connectors.

    # r = 5% everywhere
    world.paths.r .= 0.05

    # pi = 2% everywhere
    world.paths.pi .= 0.02

    # Excess Return = 10% everywhere
    world.paths.Re_Stock .= 0.10

    # ==========================================================================
    # 3. TESTS
    # ==========================================================================

    @testset "create_risk_free_return_components" begin
        # Formula: X = 1 + (r - π) * dt
        # dt = 0.5
        # X = 1 + (0.05 - 0.02) * 0.5
        # X = 1 + 0.015 = 1.015

        # Test with no outside income (O_t = nothing)
        X, Y = BrandtSolver.create_risk_free_return_components(world, 0.0, nothing)

        @test size(X) == (5, 3) # sims x (M+1)
        @test all(x -> x ≈ 1.015, X)
        @test all(y -> y == 0.0, Y)
    end

    @testset "package_excess_returns" begin
        # We request the asset named "Re_Stock"
        # The function should wrap the 0.10 value into an SVector
        Re_pkg = BrandtSolver.package_excess_returns(world, ["Re_Stock"])

        # Dimensions: (sims, steps)
        @test size(Re_pkg) == (5, 3)

        # Type: Must be a Matrix of SVectors
        @test eltype(Re_pkg) <: SVector{1, Float64}

        # Value check
        # Re_pkg[sim, time] -> SVector
        val = Re_pkg[1, 1]
        @test val[1] ≈ 0.10
    end

    @testset "get_state_variables" begin
        # Should return Vector{Matrix} corresponding to requested paths
        Z = BrandtSolver.get_state_variables(world, ["r", "pi"])

        @test length(Z) == 2
        @test Z[1] == world.paths.r
        @test Z[2] == world.paths.pi

        # Check values match our injection
        @test Z[1][1, 1] ≈ 0.05
    end
end