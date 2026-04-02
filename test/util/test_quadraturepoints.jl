using TransportMaps
using Test
using Distributions

@testset "Quadrature Points" begin
    @testset "GaussHermiteWeights" begin
        # Test basic construction
        ghw = GaussHermiteWeights(5, 2)
        @test ghw isa AbstractQuadratureWeights
        @test size(ghw.points, 1) == 25  # 5^2 points for 2D
        @test size(ghw.points, 2) == 2  # Dimension
        @test length(ghw.weights) == 25

        # Test 1D case
        ghw_1d = GaussHermiteWeights(3, 1)
        @test size(ghw_1d.points, 1) == 3
        @test size(ghw_1d.points, 2) == 1
        @test length(ghw_1d.weights) == 3

        # Test 3D case
        ghw_3d = GaussHermiteWeights(4, 3)
        @test size(ghw_3d.points, 1) == 64  # 4^3 points
        @test size(ghw_3d.points, 2) == 3
        @test length(ghw_3d.weights) == 64

        # Test weight properties (weights should be positive and finite)
        ghw_test = GaussHermiteWeights(10, 1)
        @test all(ghw_test.weights .> 0)
        @test all(isfinite.(ghw_test.weights))

        ghw_test_2d = GaussHermiteWeights(5, 2)
        @test all(ghw_test_2d.weights .> 0)
        @test all(isfinite.(ghw_test_2d.weights))

        ghw_tm = GaussHermiteWeights(2, PolynomialMap(2, 2))
        @test all(ghw_tm.weights .> 0)
        @test all(isfinite.(ghw_tm.weights))
    end

    @testset "MonteCarloWeights" begin
        # Test basic construction
        mcw = MonteCarloWeights(100, 3)
        @test mcw isa AbstractQuadratureWeights
        @test size(mcw.points, 1) == 100  # Number of points
        @test size(mcw.points, 2) == 3  # Dimension
        @test length(mcw.weights) == 100

        # Test uniform weights
        @test all(mcw.weights .≈ 1 / 100)

        # Test different dimensions
        mcw_1d = MonteCarloWeights(50, 1)
        @test size(mcw_1d.points, 1) == 50
        @test size(mcw_1d.points, 2) == 1
        @test all(mcw_1d.weights .≈ 1 / 50)

        mcw_5d = MonteCarloWeights(200, 5)
        @test size(mcw_5d.points, 1) == 200
        @test size(mcw_5d.points, 2) == 5
        @test all(mcw_5d.weights .≈ 1 / 200)

        # Test that points are different (random)
        mcw1 = MonteCarloWeights(10, 2)
        mcw2 = MonteCarloWeights(10, 2)
        @test mcw1.points != mcw2.points  # Should be different due to randomness

        mcw_tm = MonteCarloWeights(50, PolynomialMap(2, 2))
        @test all(mcw_tm.weights .> 0)

        mcw_samp = MonteCarloWeights(randn(10, 2))
        @test all(mcw_samp.weights .> 0)
    end

    @testset "LatinHypercubeWeights" begin
        # Test basic construction
        lhw = LatinHypercubeWeights(20, 2)
        @test lhw isa AbstractQuadratureWeights
        @test size(lhw.points, 1) == 20  # Number of points
        @test size(lhw.points, 2) == 2  # Dimension
        @test length(lhw.weights) == 20

        # Test uniform weights
        @test all(lhw.weights .≈ 1 / 20)

        # Test 1D case
        lhw_1d = LatinHypercubeWeights(15, 1)
        @test size(lhw_1d.points, 1) == 15
        @test size(lhw_1d.points, 2) == 1
        @test all(lhw_1d.weights .≈ 1 / 15)

        # Test higher dimensions
        lhw_4d = LatinHypercubeWeights(25, 4)
        @test size(lhw_4d.points, 1) == 25
        @test size(lhw_4d.points, 2) == 4
        @test all(lhw_4d.weights .≈ 1 / 25)

        # Test that points are distributed (Latin hypercube should give good space-filling)
        lhw_test = LatinHypercubeWeights(100, 2)
        points = lhw_test.points

        # Basic sanity checks - points should span a reasonable range
        @test minimum(points[:, 1]) < -1.0
        @test maximum(points[:, 1]) > 1.0
        @test minimum(points[:, 2]) < -1.0
        @test maximum(points[:, 2]) > 1.0

        lhw_tm = LatinHypercubeWeights(30, PolynomialMap(2, 2))
        @test all(lhw_tm.weights .> 0)
    end

    @testset "Helper Functions" begin
        # Test gausshermite_weights function directly
        points_2d, weights_2d = TransportMaps.gausshermite_weights(3, 2)
        @test size(points_2d, 1) == 9  # 3^2
        @test size(points_2d, 2) == 2
        @test length(weights_2d) == 9
        @test sum(weights_2d) ≈ 1.0 atol = 1e-10  # For normalized weights

        # Test montecarlo_weights function directly
        points_mc, weights_mc = TransportMaps.montecarlo_weights(50, 3)
        @test size(points_mc, 1) == 50
        @test size(points_mc, 2) == 3
        @test length(weights_mc) == 50
        @test all(weights_mc .≈ 1 / 50)

        # Test latinhypercube_weights function directly
        points_lh, weights_lh = TransportMaps.latinhypercube_weights(30, 2)
        @test size(points_lh, 1) == 30
        @test size(points_lh, 2) == 2
        @test length(weights_lh) == 30
        @test all(weights_lh .≈ 1 / 30)
    end

    @testset "Show" begin
        ghw = GaussHermiteWeights(3, 1)
        @test_nowarn sprint(show, ghw)
        @test_nowarn sprint(print, ghw)
        @test_nowarn display(ghw)

        mcw = MonteCarloWeights(10, 1)
        @test_nowarn sprint(show, mcw)
        @test_nowarn sprint(print, mcw)
        @test_nowarn display(mcw)

        lhw = LatinHypercubeWeights(5, 1)
        @test_nowarn sprint(show, lhw)
        @test_nowarn sprint(print, lhw)
        @test_nowarn display(lhw)

        ssw = SparseSmolyakWeights(2, 1)
        @test_nowarn sprint(show, ssw)
        @test_nowarn sprint(print, ssw)
        @test_nowarn display(ssw)
    end

    @testset "SparseSmolyakWeights" begin
        # Basic construction
        ssw = SparseSmolyakWeights(2, 2)
        @test ssw isa AbstractQuadratureWeights
        @test size(ssw.points, 2) == 2
        @test length(ssw.weights) == size(ssw.points, 1)

        # Weights should be finite; Smolyak may include negative weights so we don't
        # assert positivity here.
        @test all(isfinite.(ssw.weights))

        # Sum of weights should integrate the constant function -> 1.0 (normalized rules)
        @test sum(ssw.weights) ≈ 1.0 atol = 1e-10

        # 1D behavior: level 1 should produce 3 nodes (rule with 3 points)
        ssw1 = SparseSmolyakWeights(1, 1)
        @test size(ssw1.points, 1) == 3
        @test size(ssw1.points, 2) == 1

        # Integration sanity checks (mean ~ 0, variance ~ 1 for standard normal)
        # Use the quadrature sum: sum(f(x) * w)
        mean_est = sum(ssw.points[:, 1] .* ssw.weights)
        var_est = sum((ssw.points[:, 1] .^ 2) .* ssw.weights)
        @test abs(mean_est) < 1e-10
        @test abs(var_est - 1.0) < 1e-8

        ssw_tm = SparseSmolyakWeights(2, PolynomialMap(2, 2))
        @test all(isfinite.(ssw_tm.weights))
    end

    @testset "Integration Accuracy" begin
        # Test Gauss-Hermite quadrature accuracy with a simple Gaussian integral
        # ∫ exp(-x²) dx over ℝ = √π, but we test over a finite domain
        ghw = GaussHermiteWeights(10, 1)

        # Test integration of constant function (should equal sum of weights)
        f_const = x -> 1.0
        integral = sum(f_const(ghw.points[i]) * ghw.weights[i] for i in 1:length(ghw.weights))
        @test integral ≈ 1.0 atol = 1e-10  # Sum of normalized weights

        # Test Monte Carlo integration convergence
        function mc_integrate_gaussian_2d(n_points)
            mcw = MonteCarloWeights(n_points, 2)
            # Integrate Gaussian exp(-x²-y²) over ℝ² using samples from N(0,1)
            # This should approximate ∫∫ exp(-x²-y²) * exp(x²+y²) dx dy = ∫∫ 1 dx dy = area
            integrand_values = [exp(-sum(mcw.points[i, :] .^ 2)) for i in 1:n_points]
            return sum(integrand_values .* mcw.weights)
        end

        # This is just a basic functionality test - values should be reasonable
        result_100 = mc_integrate_gaussian_2d(100)
        result_1000 = mc_integrate_gaussian_2d(1000)

        @test 0.1 < result_100 < 2.0
        @test 0.1 < result_1000 < 2.0
    end

    @testset "Type Consistency" begin
        # Test that all quadrature methods return consistent types
        for QType in [GaussHermiteWeights, MonteCarloWeights, LatinHypercubeWeights]
            q = QType(10, 2)
            @test q.points isa Matrix{Float64}
            @test q.weights isa Vector{Float64}
            @test all(isfinite.(q.points))
            @test all(isfinite.(q.weights))
            @test all(q.weights .> 0)  # All weights should be positive
        end
    end

    @testset "Dimensions" begin
        mc = MonteCarloWeights(10, 2)
        @test numberdimensions(mc) == size(mc.points, 2)

        lhs = LatinHypercubeWeights(10, 2)
        @test numberdimensions(lhs) == size(lhs.points, 2)

        gh = GaussHermiteWeights(2, 2)
        @test numberdimensions(gh) == size(gh.points, 2)

        smo = SparseSmolyakWeights(2, 2)
        @test numberdimensions(smo) == size(smo.points, 2)
    end

    @testset "GaussLegendreWeights with U[-1,1]" begin
        # Test default constructor (U[-1,1])
        quad = GaussLegendreWeights(5, 2)

        @test size(quad.points, 1) == 25  # 5^2
        @test size(quad.points, 2) == 2
        @test length(quad.weights) == 25

        # Points should be in [-1, 1]
        @test all(-1 .<= quad.points .<= 1)

        # Sum of weights should be (b-a)^d = 2^2 = 4 for U[-1,1]^2
        @test sum(quad.weights) ≈ 4.0 atol = 1e-10
    end

    @testset "GaussLegendreWeights with U[0,1]" begin
        # Test with explicit U[0,1]
        quad = GaussLegendreWeights(5, 2, Uniform(0, 1))

        @test size(quad.points, 1) == 25
        @test size(quad.points, 2) == 2

        # Points should be in [0, 1]
        @test all(0 .<= quad.points .<= 1)

        # Sum of weights should be (b-a)^d = 1^2 = 1 for U[0,1]^2
        @test sum(quad.weights) ≈ 1.0 atol = 1e-10
    end

    @testset "GaussLegendreWeights with custom Uniform" begin
        # Test with U[2, 5]
        quad = GaussLegendreWeights(4, 1, Uniform(2, 5))

        @test size(quad.points, 1) == 4
        @test size(quad.points, 2) == 1

        # Points should be in [2, 5]
        @test all(2 .<= quad.points .<= 5)

        # Sum of weights should be (b-a)^d = 3^1 = 3 for U[2,5]
        @test sum(quad.weights) ≈ 3.0 atol = 1e-10
    end

    @testset "GaussLegendreWeights from map with U[-1,1]" begin
        # Create map with U[-1,1]
        map = PolynomialMap(2, 3, :uniform, Softplus(), LegendreBasis())
        quad = GaussLegendreWeights(5, map)

        @test size(quad.points, 2) == 2
        @test all(-1 .<= quad.points .<= 1)
        @test sum(quad.weights) ≈ 4.0 atol = 1e-10
    end

    @testset "GaussLegendreWeights from map with U[0,1]" begin
        # Create map with U[0,1]
        map = PolynomialMap(2, 3, :uniform01, Softplus(), ShiftedLegendreBasis())
        quad = GaussLegendreWeights(5, map)

        @test size(quad.points, 2) == 2
        @test all(0 .<= quad.points .<= 1)
        @test sum(quad.weights) ≈ 1.0 atol = 1e-10
    end

    @testset "GaussLegendreWeights error with non-Uniform map" begin
        # Create map with Normal reference
        map = PolynomialMap(2, 3, :normal)

        # Should error when trying to create Gauss-Legendre weights from Normal reference
        @test_throws ErrorException GaussLegendreWeights(5, map)
    end

    @testset "Integration accuracy with U[0,1]" begin
        # Test that quadrature can integrate polynomials exactly
        quad = GaussLegendreWeights(10, 1, Uniform(0, 1))

        # Integrate f(x) = x^2 over [0,1], exact value = 1/3
        f_vals = [quad.points[i, 1]^2 for i in 1:size(quad.points, 1)]
        integral = sum(quad.weights .* f_vals)

        @test integral ≈ 1 / 3 atol = 1e-10

        # Integrate f(x) = x^3 over [0,1], exact value = 1/4
        f_vals = [quad.points[i, 1]^3 for i in 1:size(quad.points, 1)]
        integral = sum(quad.weights .* f_vals)

        @test integral ≈ 1 / 4 atol = 1e-10
    end

    @testset "Integration accuracy with U[-1,1]" begin
        # Test that quadrature can integrate polynomials exactly
        quad = GaussLegendreWeights(10, 1, Uniform(-1, 1))

        # Integrate f(x) = x^2 over [-1,1], exact value = 2/3
        f_vals = [quad.points[i, 1]^2 for i in 1:size(quad.points, 1)]
        integral = sum(quad.weights .* f_vals)

        @test integral ≈ 2 / 3 atol = 1e-10

        # Integrate f(x) = 1 over [-1,1], exact value = 2
        f_vals = ones(size(quad.points, 1))
        integral = sum(quad.weights .* f_vals)

        @test integral ≈ 2.0 atol = 1e-10
    end

    @testset "Weight scaling for different dimensions" begin
        # 1D case: sum should be (b-a)^1
        quad1d = GaussLegendreWeights(5, 1, Uniform(0, 1))
        @test sum(quad1d.weights) ≈ 1.0 atol = 1e-10

        # 2D case: sum should be (b-a)^2
        quad2d = GaussLegendreWeights(5, 2, Uniform(0, 1))
        @test sum(quad2d.weights) ≈ 1.0 atol = 1e-10

        # 3D case: sum should be (b-a)^3
        quad3d = GaussLegendreWeights(5, 3, Uniform(0, 1))
        @test sum(quad3d.weights) ≈ 1.0 atol = 1e-10
    end

end
