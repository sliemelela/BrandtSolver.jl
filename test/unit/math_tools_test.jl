using BrandtSolver
using LinearAlgebra
using Test

@testset "Utils: Math Kernels" begin
    # 1. Automatic Differentiation Wrapper
    @testset "create_utility_from_ad" begin
        # Test with a simple function: u(W) = W^3
        # u'(W) = 3W^2, u''(W) = 6W
        simple_u = W -> W^3
        utils = create_utility_from_ad(simple_u)

        W_val = 2.0
        @test utils.nth_derivative(1)(W_val) ≈ 12.0 # 3 * 2^2
        @test utils.nth_derivative(2)(W_val) ≈ 12.0  # 6 * 2
    end

    # 2. Regression Matrix Construction
    @testset "power_matrix" begin
        # Z has 2 vars, 3 observations
        Z1 = [1.0, 2.0, 3.0]
        Z2 = [4.0, 5.0, 6.0]
        Z = [Z1, Z2]
        p = 2

        # Expected: [Intercept, Z1^1, Z1^2, Z2^1, Z2^2]
        Phi = BrandtSolver.power_matrix(Z, p)

        @test size(Phi) == (3, 5) # 3 sims, 1 + 2*2 cols
        @test Phi[:, 1] == ones(3) # Intercept
        @test Phi[:, 3] == [1.0, 4.0, 9.0] # Z1^2
        @test Phi[:, 5] == [16.0, 25.0, 36.0] # Z2^2
    end

    # 3. OLS Estimation
    @testset "estimate_coefficients" begin
        # Create perfect linear relationship: Y = 2 + 3*X
        X = [1.0, 2.0, 3.0, 4.0]
        Y = 2.0 .+ 3.0 .* X

        # Design matrix: [1, X]
        Phi = hcat(ones(4), X)
        qrPhi = qr(Phi)

        α = StandardOLS()
        theta = BrandtSolver.estimate_coefficients(α, qrPhi, Y)
        @test theta[1] ≈ 2.0 atol=1e-10
        @test theta[2] ≈ 3.0 atol=1e-10
    end
end