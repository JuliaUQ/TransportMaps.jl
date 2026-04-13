# This file will contain tests for general MultivariateBasis logic, not specific to Hermite
using TransportMaps
using Test

@testset "MultivariateBasis General" begin
    @testset "Multi-index generation" begin
        idx = multivariate_indices(2, 2)
        @test length(idx) > 0
        @test all(length(i) == 2 for i in idx)
    end

    @testset "Show" begin
        mb = MultivariateBasis([1, 2], HermiteBasis())
        @test_nowarn sprint(show, mb)
        @test_nowarn sprint(print, mb)
        @test_nowarn display(mb)
    end

end

@testset "MultivariateBasis - Hermite specific" begin
    # Test multivariate Psi (using HermiteBasis)
    alpha = [0, 1]
    x = [1.0, 2.0]
    result = Psi(alpha, x, [HermiteBasis(), HermiteBasis()])
    expected = basisfunction(HermiteBasis(), 0, 1.0) * basisfunction(HermiteBasis(), 1, 2.0)
    @test result ≈ expected

    # Test with different dimensions
    alpha3 = [1, 0, 2]
    x3 = [0.5, 1.0, -0.5]
    result3 = Psi(alpha3, x3, [HermiteBasis(), HermiteBasis(), HermiteBasis()])
    expected3 = basisfunction(HermiteBasis(), 1, 0.5) * basisfunction(HermiteBasis(), 0, 1.0) * basisfunction(HermiteBasis(), 2, -0.5)
    @test result3 ≈ expected3

    # Test MultivariateBasis creation
    mvb = MultivariateBasis([1, 2, 0], HermiteBasis())
    @test mvb.multiindexset == [1, 2, 0]
    @test basistype(mvb) == HermiteBasis

    # Test evaluation (default Hermite)
    x = [1.0, 0.0, 2.0]
    result = evaluate(mvb, x)
    expected = hermite_polynomial(1, 1.0) * hermite_polynomial(2, 0.0) * hermite_polynomial(0, 2.0)
    @test result ≈ expected

    # Test evaluation with edge control
    mvb_gauss = MultivariateBasis([1, 2, 0], GaussianWeightedHermiteBasis())
    result_gauss = evaluate(mvb_gauss, x)
    expected_gauss = basisfunction(GaussianWeightedHermiteBasis(), 1, 1.0) * basisfunction(GaussianWeightedHermiteBasis(), 2, 0.0) * basisfunction(GaussianWeightedHermiteBasis(), 0, 2.0)
    @test result_gauss ≈ expected_gauss

    # Create basis functions
    mvb1 = MultivariateBasis([0, 0], HermiteBasis())  # constant
    mvb2 = MultivariateBasis([1, 0], HermiteBasis())  # x1
    mvb3 = MultivariateBasis([0, 1], HermiteBasis())  # x2

    Psi_vec = [mvb1, mvb2, mvb3]
    coefficients = [1.0, 2.0, 3.0]  # f = 1 + 2*x1 + 3*x2

    x = [1.0, 2.0]
    result = f(Psi_vec, coefficients, x)
    expected = 1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 2.0  # 1 + 2 + 6 = 9
    @test result ≈ expected

    # Test functional interface
    func = f(Psi_vec, coefficients)
    @test func(x) ≈ expected

    # Test Hermite polynomial derivatives
    @test hermite_derivative(0, 1.0) ≈ 0.0
    @test hermite_derivative(1, 1.0) ≈ 1.0
    @test hermite_derivative(2, 0.0) ≈ 0.0  # H_2'(x) = 2*H_1(x) = 2*x, at x=0: 2*0 = 0
    @test hermite_derivative(3, 1.0) ≈ 0.0  # H_3'(x) = 3*H_2(x) = 3*(x^2-1), at x=1: 3*(1-1) = 0

    # Verify the derivative formula: H_n'(x) = n*H_{n-1}(x)
    @test hermite_derivative(3, 1.0) ≈ 3.0 * hermite_polynomial(2, 1.0)

    # Test partial derivatives of MultivariateBasis
    mvb = MultivariateBasis([1, 2], HermiteBasis())
    x = [1.0, 0.5]

    pd1 = partial_derivative_z(mvb, x, 1)
    expected1 = hermite_derivative(1, x[1]) * hermite_polynomial(2, x[2])
    @test pd1 ≈ expected1

    pd2 = partial_derivative_z(mvb, x, 2)
    expected2 = hermite_polynomial(1, x[1]) * hermite_derivative(2, x[2])
    @test pd2 ≈ expected2

    # Test gradient
    grad = gradient_z(mvb, x)
    @test length(grad) == 2
    @test grad[1] ≈ pd1
    @test grad[2] ≈ pd2

    # Create a simple function f = 2*x1 + 3*x2
    mvb1 = MultivariateBasis([1, 0], HermiteBasis())  # x1
    mvb2 = MultivariateBasis([0, 1], HermiteBasis())  # x2

    Psi_vec = [mvb1, mvb2]
    coefficients = [2.0, 3.0]
    x = [1.0, 2.0]

    grad_z = gradient_z(Psi_vec, coefficients, x)
    @test grad_z ≈ [2.0, 3.0]

    grad_c = gradient_coefficients(Psi_vec, x)
    expected_grad_c = [evaluate(mvb1, x), evaluate(mvb2, x)]
    @test grad_c ≈ expected_grad_c

    # Test constructor for linearized hermite basis within multivariate basis
    mvb = MultivariateBasis([1, 2], LinearizedHermiteBasis)
    @test mvb.multiindexset == [1, 2]
    @test mvb.univariatebases[1] isa LinearizedHermiteBasis
    @test mvb.univariatebases[2] isa LinearizedHermiteBasis
end
