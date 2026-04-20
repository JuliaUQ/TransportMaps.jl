using TransportMaps
using Test
using Distributions

@testset "QuadratureWeights" begin

    @testset "TensorProductWeights" begin

        @testset "GaussHermite" begin
            tp_gh = TensorProductWeights(2, 2, GaussHermiteKnots())
            @test isa(tp_gh, TensorProductWeights{GaussHermiteKnots})
            n = 2^2 + 1
            @test size(tp_gh.points, 1) == n^2  # n^d points for d=2
            @test size(tp_gh.points, 2) == 2
            @test length(tp_gh.weights) == n^2

            @test all(tp_gh.weights .> 0)
            @test all(isfinite.(tp_gh.weights))

            # GaussHermiteWeights constructor
            gh = GaussHermiteWeights(2, 2)
            @test isa(gh, TensorProductWeights{GaussHermiteKnots})
            @test size(gh.points, 1) == n^2  # n^d points for d=2
            @test size(gh.points, 2) == 2
            @test length(gh.weights) == n^2

            # Knots from map
            P = PolynomialMap(2, 2) # Normal reference
            weights = TensorProductWeights(2, P) # Normal
            @test eltype(weights) == GaussHermiteKnots

            # Knots from map with GaussHermiteWeights constructor
            gh_map = GaussHermiteWeights(2, P)
            @test size(gh_map.points, 1) == n^2  # n^d points for d=2
            @test size(gh_map.points, 2) == 2
            @test length(gh_map.weights) == n^2

            # Knots from map with uniform density
            P_uniform = PolynomialMap(2, 2, Uniform(), Softplus(), ShiftedLegendreBasis())
            @test_throws "GaussHermiteWeights requires Normal" GaussHermiteWeights(2, P_uniform)
        end

        @testset "GaussLegendre" begin
            tp_gl = TensorProductWeights(2, 2, GaussLegendreKnots())
            @test isa(tp_gl, TensorProductWeights{GaussLegendreKnots})
            n = 2^2 + 1
            @test size(tp_gl.points, 1) == n^2  # n^d points for d=2
            @test size(tp_gl.points, 2) == 2
            @test length(tp_gl.weights) == n^2

            @test all(tp_gl.weights .> 0)
            @test all(isfinite.(tp_gl.weights))

            # GaussLegendreWeights constructor
            gl = GaussLegendreWeights(2, 2)
            @test isa(gl, TensorProductWeights{GaussLegendreKnots})
            @test size(gl.points, 1) == n^2  # n^d points for d=2
            @test size(gl.points, 2) == 2
            @test length(gl.weights) == n^2

            # Knots from map
            P_uniform = PolynomialMap(2, 2, Uniform(), Softplus(), ShiftedLegendreBasis())
            weights = GaussLegendreWeights(2, P_uniform) # Normal
            @test eltype(weights) == GaussLegendreKnots

            # Knots from map with GaussLegendreWeights constructor
            gh_map = GaussLegendreWeights(2, P_uniform)
            @test size(gh_map.points, 1) == n^2  # n^d points for d=2
            @test size(gh_map.points, 2) == 2
            @test length(gh_map.weights) == n^2

            # Knots from map with normal density
            P_normal = PolynomialMap(2, 2, Normal())
            @test_throws "GaussLegendreWeights requires Uniform" GaussLegendreWeights(2, P_normal)

        end

    end

    @testset "SparseSmolyakWeights" begin

        sparse = SparseSmolyakWeights(2, 2)
        @test isa(sparse, SparseSmolyakWeights{GaussHermiteKnots})
        @test size(sparse.points, 2) == 2
        @test length(sparse.weights) == size(sparse.points, 1)
        @test sum(sparse.weights) ≈ 1.0 atol = 1e-10

        sparse_tm = SparseSmolyakWeights(2, PolynomialMap(2, 2))
        @test isa(sparse_tm, SparseSmolyakWeights{GaussHermiteKnots})

        sparse_legendre = SparseSmolyakWeights(2, 2, GaussLegendreKnots())
        @test isa(sparse_legendre, SparseSmolyakWeights{GaussLegendreKnots})

        sparse_tm_2 = SparseSmolyakWeights(2, PolynomialMap(1, 2, Uniform(), Softplus(), ShiftedLegendreBasis()))
        @test isa(sparse_legendre, SparseSmolyakWeights{GaussLegendreKnots})

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

        mcw_tm = MonteCarloWeights(30, PolynomialMap(2, 2))
        @test all(mcw_tm.weights .> 0)
        @test isa(mcw_tm.distribution, Normal)

        mcw_tm = MonteCarloWeights(30, PolynomialMap(2, 2, Uniform(), Softplus(), ShiftedLegendreBasis()))
        @test all(mcw_tm.weights .> 0)
        @test isa(mcw_tm.distribution, Uniform)

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
        @test isa(lhw_tm.distribution, Normal)

        lhw_tm = LatinHypercubeWeights(30, PolynomialMap(2, 2, Uniform(), Softplus(), ShiftedLegendreBasis()))
        @test all(lhw_tm.weights .> 0)
        @test isa(lhw_tm.distribution, Uniform)
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


end
