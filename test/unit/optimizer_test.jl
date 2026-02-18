using BrandtSolver
using Statistics
using Test

# @testset "Merton Benchmark" begin
#     # Setup
#     sim = 10_000
#     γ = 5.0
#     μ_excess = 0.05
#     σ = 0.20
#     W_t = 1.0

#     # Theory: 0.05 / (5 * 0.04) = 0.25
#     ω_target = μ_excess / (γ * σ^2)

#     # Create Fake Inputs for calculate_optimal_policy
#     # E[a] ≈ μ
#     # E[b] ≈ -γ * (σ²)
#     E_a = fill(μ_excess, sim, 1)
#     val_b = -γ * (σ^2)
#     E_b = fill(val_b, sim, 1, 1)

#     # Run Core Function
#     ω_svec = BrandtSolver.calculate_optimal_policy(E_a, E_b, W_t)
#     ω_computed = mean([w[1] for w in ω_svec])

#     # We allow 1% tolerance for numerical floating point noise
#     @test isapprox(ω_computed, ω_target, rtol=0.01)
# end

# @testset "Identical Assets" begin
#     println("\n--- Testing Singularity Robustness ---")
#     sim = 100
#     N_assets = 2

#     # 1. Inputs
#     # Both assets have identical attractive returns
#     E_a = fill(0.05, sim, N_assets)

#     # Matrix E[b] implies perfect correlation
#     # [ -1  -1 ]
#     # [ -1  -1 ]
#     # This matrix has eigenvalues [ -2, 0 ]. It is non-invertible.
#     E_b = zeros(sim, N_assets, N_assets)
#     for i in 1:sim
#         E_b[i, :, :] .= -1.0
#     end
#     W_t = 1.0

#     # 2. Run Solver
#     # This calls your function which has `jit = 1e-10 * I`
#     ω_svec = BrandtSolver.calculate_optimal_policy(E_a, E_b, W_t)

#     # 3. Analyze Results
#     ω_mean = mean(ω_svec)
#     println("  > Weights computed: $ω_mean")

#     # CHECK A: Did it explode?
#     # If regularization is too weak, these might be 1e12 or NaN
#     total_exposure = sum(abs.(ω_mean))
#     println("  > Total Absolute Exposure: $total_exposure")

#     # We expect the total weight to be roughly equal to the single asset case (0.05)
#     # But split across two assets.
#     @test total_exposure < 10.0

#     # CHECK B: Symmetry
#     # Since assets are identical, the solver should treat them identically.
#     # w1 should approx w2
#     diff = abs(ω_mean[1] - ω_mean[2])
#     println("  > Asymmetry (w1 - w2): $diff")

#     @test diff < 0.1
# end

# If an asset has + return and 0 variance, the optimizer should explode (go to infinity), or hit
# numerical limits.
# This tests if your code handles extreme numbers without crashing.
# @testset "Zero Variance Safety" begin
#     sim = 10
#     E_a = fill(0.05, sim, 1) # Positive return
#     E_b = fill(-1e-10, sim, 1, 1) # Tiny risk (approaching zero)
#     W_t = 1.0

#     # Should not throw error
#     ω_svec = calculate_optimal_policy(E_a, E_b, W_t)

#     # Result should be huge
#     @test mean(ω_svec)[1] > 100.0
# end