@testsnippet HybridRootFinderSetup begin
    using TransportMaps
    using Test
    using Logging

end

@testitem "Hybrid root finder at floating-point precision" setup = [HybridRootFinderSetup] begin
    @testset "Inclusive residual tolerance" begin
        calls = Ref(0)
        f(x) = (calls[] += 1; x - 1.0)
        tolerance = 2eps(Float64)
        root, residual, _ = @test_logs min_level = Logging.Warn hybridrootfinder(
            f, x -> 1.0, 1.0, 1.0 + 4eps(Float64);
            xtol = 0.0, ftol = tolerance,
        )
        @test abs(residual) <= tolerance
        @test abs(root - 1.0) <= tolerance
        @test calls[] == 1
    end

    @testset "Exact root with zero derivative and zero tolerances" begin
        result = @test_logs min_level = Logging.Warn hybridrootfinder(
            x -> x^3, x -> 3x^2, -1.0, 1.0; xtol = 0.0, ftol = 0.0,
        )
        @test result == (0.0, 0.0, 0.0)
    end

    @testset "Newton step reaches an endpoint root" begin
        calls = Ref(0)
        f(x) = (calls[] += 1; x - 1.0)
        root, residual, _ = @test_logs min_level = Logging.Warn hybridrootfinder(
            f, x -> 1.0, -1.0, 1.0,
        )
        @test root == 1.0
        @test residual == 0.0
        @test calls[] == 2
    end

    @testset "Bracket exhausted at adjacent floats" begin
        calls = Ref(0)
        f(x) = (calls[] += 1; x^2 - 2.0)
        root, residual, derivative = @test_logs min_level = Logging.Warn hybridrootfinder(
            f, x -> 2x, 1.0, 2.0; xtol = 0.0, ftol = 0.0,
        )
        @test abs(root - sqrt(2.0)) <= eps(sqrt(2.0))
        @test abs(residual) <= 2eps(2.0)
        @test derivative == 2root
        @test calls[] < 100
    end

    @testset "Unsafe Newton steps fall back to bisection" begin
        calls = Ref(0)
        function f(x)
            -1000.0 <= x <= 2500.0 || error("Root iteration left its bracket")
            calls[] += 1
            return expm1(x)
        end
        # The initial residual and derivative overflow. Subsequent Newton steps
        # can leave the bracket or make very little progress in its steep tail.
        root, residual, _ = @test_logs min_level = Logging.Warn hybridrootfinder(
            f, exp, -1000.0, 2500.0; xtol = 1.0e-12, ftol = 1.0e-12, maxiter = 100,
        )
        @test abs(root) <= 1.0e-12
        @test abs(residual) <= 1.0e-12
        @test calls[] < 100
    end
end

@testitem "Hybrid Root Finder" setup = [HybridRootFinderSetup] begin
    @testset "_inverse_bound Function" begin
        # Test with simple linear function
        f_linear(x) = x - 1.0  # Root at x = 1
        lower, upper = TransportMaps._inverse_bound(f_linear)
        @test f_linear(lower) * f_linear(upper) <= 0.0  # Root should be bracketed
        @test lower < upper

        # Test with quadratic function
        f_quad(x) = x^2 - 4.0  # Roots at x = ±2
        lower, upper = TransportMaps._inverse_bound(f_quad)
        @test f_quad(lower) * f_quad(upper) <= 0.0  # Root should be bracketed
        @test lower < upper

        # Test with function that has root at 0
        f_zero(x) = x  # Root at x = 0
        lower, upper = TransportMaps._inverse_bound(f_zero)
        @test f_zero(lower) * f_zero(upper) <= 0.0  # Root should be bracketed
        @test lower <= 0.0 <= upper
    end

    @testset "hybridrootfinder Function - Basic Cases" begin
        # Test with simple linear function
        f_linear(x) = x - 2.0  # Root at x = 2
        ∂f_linear(x) = 1.0    # Derivative is constant 1

        root, fval, dfval = hybridrootfinder(f_linear, ∂f_linear, 0.0, 5.0)
        @test root ≈ 2.0 atol = 1.0e-6
        @test abs(fval) < 1.0e-6
        @test dfval ≈ 1.0

        # Test with quadratic function (positive root)
        f_quad(x) = x^2 - 9.0  # Roots at x = ±3
        ∂f_quad(x) = 2 * x

        # Find positive root
        root, fval, dfval = hybridrootfinder(f_quad, ∂f_quad, 0.0, 5.0)
        @test root ≈ 3.0 atol = 1.0e-6
        @test abs(fval) < 1.0e-6
        @test dfval ≈ 6.0 atol = 1.0e-6

        # Test with cubic function
        f_cubic(x) = x^3 - 8.0  # Root at x = 2
        ∂f_cubic(x) = 3 * x^2

        root, fval, dfval = hybridrootfinder(f_cubic, ∂f_cubic, 1.0, 3.0)
        @test root ≈ 2.0 atol = 1.0e-6
        @test abs(fval) < 1.0e-6
        @test dfval ≈ 12.0 atol = 1.0e-6
    end

    @testset "Robust Root Finding" begin
        # Test with exponential function: e^x - 2 = 0, root at x = ln(2)
        f_exp(x) = exp(x) - 2.0
        ∂f_exp(x) = exp(x)

        root, fval, dfval = hybridrootfinder(f_exp, ∂f_exp, 0.0, 1.0)
        @test root ≈ log(2.0) atol = 1.0e-6
        @test abs(fval) < 1.0e-6

        # Test with function where root is at boundary
        f_boundary(x) = x  # Root at x = 0
        ∂f_boundary(x) = 1.0

        root, fval, dfval = hybridrootfinder(f_boundary, ∂f_boundary, -1.0, 1.0)
        @test root ≈ 0.0 atol = 1.0e-6
        @test abs(fval) < 1.0e-6

        # Test with tight tolerance
        f_test(x) = x^2 - 4.0
        ∂f_test(x) = 2 * x

        root, fval, dfval = hybridrootfinder(f_test, ∂f_test, 1.0, 3.0, ftol = 1.0e-12, xtol = 1.0e-12)
        @test root ≈ 2.0 atol = 1.0e-10
        @test abs(fval) < 1.0e-12
    end

    @testset "Convergence Properties" begin
        # Test convergence with different initial bounds
        f_convergence(x) = x^3 - x - 1.0  # Has root around x ≈ 1.324717957...
        ∂f_convergence(x) = 3 * x^2 - 1.0

        # Test with wide bounds
        root1, _, _ = hybridrootfinder(f_convergence, ∂f_convergence, 0.0, 10.0)
        # Test with narrow bounds
        root2, _, _ = hybridrootfinder(f_convergence, ∂f_convergence, 1.0, 2.0)

        @test abs(root1 - root2) < 1.0e-6  # Should converge to same root
        @test abs(f_convergence(root1)) < 1.0e-6
        @test abs(f_convergence(root2)) < 1.0e-6

        # Test function returning proper number of values
        root, fval, dfval = hybridrootfinder(f_convergence, ∂f_convergence, 1.0, 2.0)
        @test isa(root, Real)
        @test isa(fval, Real)
        @test isa(dfval, Real)
    end

    @testset "Warning" begin
        # An insufficient iteration budget must still produce a warning.
        f_flat(x) = x^3 - 3x + 2  # Has root at x = 1
        ∂f_flat(x) = 3x^2 - 3

        @test_logs (:warn, "Maximum iterations reached in hybridrootfinder") hybridrootfinder(
            f_flat, ∂f_flat, 1.0, 3.0; maxiter = 5,
        )
    end
end
