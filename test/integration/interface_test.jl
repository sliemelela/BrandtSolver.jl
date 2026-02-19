using BrandtSolver
using FinancialMarketSimulation
using StaticArrays
using Test

include("../test_helpers.jl")

@testset "Interface Integration Tests" begin
    # 1. SETUP: Mocking the World
    r_proc = VasicekProcess(:r, 0.1, 0.05, 0.01, 0.05, 1)
    pi_proc = VasicekProcess(:pi, 0.1, 0.02, 0.01, 0.02, 2)
    stock_proc = GenericSDEProcess(:S, (t,x)->0.0, (t,x)->0.0, 100.0, [3])
    re_proc = ExcessReturnProcess(:Re_Stock, :S, :r)

    config = MarketConfig(sims = 5, T = 1.0, M = 2, processes = [r_proc, pi_proc, stock_proc, re_proc])
    world = build_world(config)

    # 2. INJECTION: Overwrite with Deterministic Data
    world.paths.r .= 0.05
    world.paths.pi .= 0.02
    world.paths.Re_Stock .= 0.10

    # 3. TESTS
    @testset "create_risk_free_return_components" begin
        X, Y = create_risk_free_return_components(world, 0.0, nothing)
        @test size(X) == (5, 3)
        @test all(x -> x ≈ 1.015, X)
        @test all(y -> y == 0.0, Y)
    end

    @testset "package_excess_returns" begin
        Re_pkg = package_excess_returns(world, ["Re_Stock"])
        @test size(Re_pkg) == (5, 3)
        @test eltype(Re_pkg) <: SVector{1, Float64}
        @test Re_pkg[1, 1][1] ≈ 0.10
    end

    @testset "get_state_variables" begin
        Z = get_state_variables(world, ["r", "pi"])
        @test length(Z) == 2
        @test Z[1] == world.paths.r
        @test Z[2] == world.paths.pi
        @test Z[1][1, 1] ≈ 0.05
    end
end