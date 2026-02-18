using BrandtSolver
using StaticArrays
using Test


@testset "calculate_next_wealth" begin
    W_curr = [100.0, 100.0]
    w_pol = [SVector(0.5, 0.5), SVector(0.0, 1.0)] # 50/50 and 0/100
    Re_next = [SVector(0.1, 0.2), SVector(0.1, 0.2)] # Asset returns
    R_free = [1.02, 1.02] # Risk free gross return

    # Sim 1: 100 * ( (0.5*0.1 + 0.5*0.2) + 1.02 ) = 100 * (0.15 + 1.02) = 117
    # Sim 2: 100 * ( (0*0.1 + 1*0.2) + 1.02 ) = 100 * (0.2 + 1.02) = 122

    W_next = BrandtSolver.calculate_next_wealth(W_curr, w_pol, Re_next, R_free)
    @test W_next[1] ≈ 117.0
    @test W_next[2] ≈ 122.0
end