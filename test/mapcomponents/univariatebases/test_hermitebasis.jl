using TransportMaps
using Test
using Statistics


@testset "HermiteBasis" begin
    # Test first few Hermite polynomials at x = 0
    @test hermite_polynomial(0, 0.0) ≈ 1.0
    @test hermite_polynomial(1, 0.0) ≈ 0.0
    @test hermite_polynomial(2, 0.0) ≈ -1.0
    @test hermite_polynomial(3, 0.0) ≈ 0.0

    # Test at x = 1
    @test hermite_polynomial(0, 1.0) ≈ 1.0
    @test hermite_polynomial(1, 1.0) ≈ 1.0
    @test hermite_polynomial(2, 1.0) ≈ 0.0  # x^2 - 1 at x=1
    @test hermite_polynomial(3, 1.0) ≈ -2.0  # x^3 - 3x at x=1

    # Test basisfunction interface
    hb = HermiteBasis()
    @test basisfunction(hb, 0, 1.0) ≈ 1.0
    @test basisfunction(hb, 1, 1.0) ≈ 1.0
    @test basisfunction(hb, 2, 0.0) ≈ -1.0

    # Derivative tests
    @test hermite_derivative(0, 1.0) ≈ 0.0
    @test hermite_derivative(1, 1.0) ≈ 1.0
    @test hermite_derivative(2, 1.0) ≈ 2.0

    # Basis function derivative tests
    @test basisfunction_derivative(hb, 0, 1.0) ≈ 0.0
    @test basisfunction_derivative(hb, 1, 1.0) ≈ 1.0
    @test basisfunction_derivative(hb, 2, 1.0) ≈ 2.0

    @test_throws MethodError HermiteBasis(1)

    @testset "Show" begin
        hb_show = HermiteBasis()
        @test_nowarn sprint(show, hb_show)
        @test_nowarn sprint(print, hb_show)
        @test_nowarn display(hb_show)
    end
end
