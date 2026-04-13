using Test
using TransportMaps
using Distributions

@testset "LegendreBasis" begin
    basis = LegendreBasis()

    @testset "Polynomial evaluation" begin
        # Test P_0(x) = 1
        @test legendre_polynomial(0, 0.0) ≈ 1.0
        @test legendre_polynomial(0, 0.5) ≈ 1.0
        @test legendre_polynomial(0, -0.5) ≈ 1.0

        # Test P_1(x) = x
        @test legendre_polynomial(1, 0.0) ≈ 0.0
        @test legendre_polynomial(1, 0.5) ≈ 0.5
        @test legendre_polynomial(1, -0.5) ≈ -0.5

        # Test P_2(x) = (3x^2 - 1)/2
        @test legendre_polynomial(2, 0.0) ≈ -0.5
        @test legendre_polynomial(2, 0.5) ≈ (3*0.5^2 - 1)/2
        @test legendre_polynomial(2, 1.0) ≈ 1.0

        # Test P_3(x) = (5x^3 - 3x)/2
        @test legendre_polynomial(3, 0.0) ≈ 0.0
        @test legendre_polynomial(3, 1.0) ≈ 1.0
        @test legendre_polynomial(3, -1.0) ≈ -1.0
    end

    @testset "Derivative evaluation" begin
        # Test P'_0(x) = 0
        @test legendre_derivative(0, 0.0) ≈ 0.0
        @test legendre_derivative(0, 0.5) ≈ 0.0

        # Test P'_1(x) = 1
        @test legendre_derivative(1, 0.0) ≈ 1.0
        @test legendre_derivative(1, 0.5) ≈ 1.0

        # Test P'_2(x) = 3x
        @test legendre_derivative(2, 0.0) ≈ 0.0
        @test legendre_derivative(2, 0.5) ≈ 1.5
        @test legendre_derivative(2, 1.0) ≈ 3.0
    end

    @testset "Basis function interface" begin
        # Test basisfunction
        @test basisfunction(basis, 0, 0.5) ≈ legendre_polynomial(0, 0.5)
        @test basisfunction(basis, 1, 0.5) ≈ legendre_polynomial(1, 0.5)
        @test basisfunction(basis, 2, 0.5) ≈ legendre_polynomial(2, 0.5)

        # Test basisfunction_derivative
        @test basisfunction_derivative(basis, 0, 0.5) ≈ legendre_derivative(0, 0.5)
        @test basisfunction_derivative(basis, 1, 0.5) ≈ legendre_derivative(1, 0.5)
        @test basisfunction_derivative(basis, 2, 0.5) ≈ legendre_derivative(2, 0.5)
    end

    @testset "Orthogonality (numerical check)" begin
        # Legendre polynomials should be orthogonal on [-1, 1]
        # We'll use simple numerical integration (trapezoidal rule)
        
        # Test orthogonality of P_0 and P_1
        x = range(-1, 1, length=1000)
        integral = sum(legendre_polynomial(0, xi) * legendre_polynomial(1, xi) for xi in x) * (x[2] - x[1])
        @test abs(integral) < 1e-2
        
        # Test orthogonality of P_1 and P_2
        integral = sum(legendre_polynomial(1, xi) * legendre_polynomial(2, xi) for xi in x) * (x[2] - x[1])
        @test abs(integral) < 1e-2
        
        # Test norm of P_0: ∫_{-1}^{1} P_0^2 dx = 2
        integral = sum(legendre_polynomial(0, xi)^2 for xi in x) * (x[2] - x[1])
        @test abs(integral - 2.0) < 1e-2
        
        # Test norm of P_1: ∫_{-1}^{1} P_1^2 dx = 2/3
        integral = sum(legendre_polynomial(1, xi)^2 for xi in x) * (x[2] - x[1])
        @test abs(integral - 2/3) < 1e-2
    end
end

@testset "ShiftedLegendreBasis" begin
    basis = ShiftedLegendreBasis()
    
    @testset "Polynomial evaluation" begin
        # Test P_0^*(x) = 1
        @test shifted_legendre_polynomial(0, 0.0) ≈ 1.0
        @test shifted_legendre_polynomial(0, 0.5) ≈ 1.0
        @test shifted_legendre_polynomial(0, 1.0) ≈ 1.0
        
        # Test P_1^*(x) = 2x - 1
        @test shifted_legendre_polynomial(1, 0.0) ≈ -1.0
        @test shifted_legendre_polynomial(1, 0.5) ≈ 0.0
        @test shifted_legendre_polynomial(1, 1.0) ≈ 1.0
        
        # Test transformation property: P_n^*(x) = P_n(2x-1)
        @test shifted_legendre_polynomial(2, 0.5) ≈ legendre_polynomial(2, 0.0)
        @test shifted_legendre_polynomial(2, 0.75) ≈ legendre_polynomial(2, 0.5)
        @test shifted_legendre_polynomial(3, 0.25) ≈ legendre_polynomial(3, -0.5)
    end
    
    @testset "Derivative evaluation" begin
        # Test P_0^*' = 0
        @test shifted_legendre_derivative(0, 0.0) ≈ 0.0
        @test shifted_legendre_derivative(0, 0.5) ≈ 0.0
        @test shifted_legendre_derivative(0, 1.0) ≈ 0.0
        
        # Test P_1^*' = 2 (derivative of 2x-1)
        @test shifted_legendre_derivative(1, 0.0) ≈ 2.0
        @test shifted_legendre_derivative(1, 0.5) ≈ 2.0
        @test shifted_legendre_derivative(1, 1.0) ≈ 2.0
        
        # Test chain rule: d/dx P_n(2x-1) = 2 * P_n'(2x-1)
        x = 0.6
        @test shifted_legendre_derivative(2, x) ≈ 2 * legendre_derivative(2, 2*x - 1)
    end
    
    @testset "Basis function interface" begin
        # Test basisfunction
        @test basisfunction(basis, 0, 0.5) ≈ shifted_legendre_polynomial(0, 0.5)
        @test basisfunction(basis, 1, 0.5) ≈ shifted_legendre_polynomial(1, 0.5)
        @test basisfunction(basis, 2, 0.25) ≈ shifted_legendre_polynomial(2, 0.25)
        
        # Test basisfunction_derivative
        @test basisfunction_derivative(basis, 0, 0.5) ≈ shifted_legendre_derivative(0, 0.5)
        @test basisfunction_derivative(basis, 1, 0.5) ≈ shifted_legendre_derivative(1, 0.5)
        @test basisfunction_derivative(basis, 2, 0.75) ≈ shifted_legendre_derivative(2, 0.75)
    end
    
    @testset "Orthogonality on [0,1]" begin
        # Shifted Legendre polynomials should be orthogonal on [0, 1]
        x = range(0, 1, length=1000)
        dx = x[2] - x[1]
        
        # Test orthogonality of P_0^* and P_1^*
        integral = sum(shifted_legendre_polynomial(0, xi) * shifted_legendre_polynomial(1, xi) for xi in x) * dx
        @test abs(integral) < 1e-2
        
        # Test orthogonality of P_1^* and P_2^*
        integral = sum(shifted_legendre_polynomial(1, xi) * shifted_legendre_polynomial(2, xi) for xi in x) * dx
        @test abs(integral) < 1e-2
        
        # Test norm: ∫_0^1 (P_0^*)^2 dx = 1
        integral = sum(shifted_legendre_polynomial(0, xi)^2 for xi in x) * dx
        @test abs(integral - 1.0) < 1e-2
        
        # Test norm: ∫_0^1 (P_1^*)^2 dx = 1/3
        integral = sum(shifted_legendre_polynomial(1, xi)^2 for xi in x) * dx
        @test abs(integral - 1/3) < 1e-2
    end
end
