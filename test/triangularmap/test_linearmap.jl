@testsnippet LinearMapSetup begin
    using TransportMaps
    using Test
    using LinearAlgebra
    using Random

end

@testitem "LinearMap distribution" setup = [LinearMapSetup] begin
    using Distributions, Statistics

    L = LinearMap([1.0, -2.0], [2.0, 3.0])
    d = TransportMapDistribution(L)
    expected = MvNormal(L.μ, Diagonal(L.σ .^ 2))
    X = [0.0 0.0; 1.0 -2.0; -0.5 0.3]
    @test length(d) == 2
    @test mean(d) ≈ mean(expected)
    @test var(d) ≈ var(expected)
    @test std(d) ≈ std(expected)
    @test cov(d) ≈ cov(expected)
    @test cor(d) ≈ cor(expected)
    @test entropy(d) ≈ entropy(expected)
    @test logpullback(L, X) ≈ logpdf(expected, permutedims(X))
    @test pdf(d, permutedims(X)) ≈ pdf(expected, permutedims(X))
    @test loglikelihood(d, permutedims(X)) ≈ loglikelihood(expected, permutedims(X))
    @test logabsdetjacobian(L, X) ≈ fill(-log(6), 3)
    @test jacobian(L, X) ≈ fill(1 / 6, 3)
    @test insupport(d, X') == trues(3)
    @test !insupport(d, [Inf, 0.0])
    @test logpdf(d, [Inf, 0.0]) == -Inf
    @test isempty(logpdf(d, zeros(2, 0)))
    @test_throws DimensionMismatch logpdf(d, zeros(3))

    rng = MersenneTwister(101)
    expected_point = inverse(L, rand(MersenneTwister(101), MvNormal(zeros(2), I)))
    point = fill(NaN, 2)
    @test rand!(rng, d, point) === point
    @test point ≈ expected_point
    @test size(rand(MersenneTwister(1), d, 7)) == (2, 7)
    output = fill(NaN, 2, 5)
    expected_samples = inverse(L, permutedims(rand(MersenneTwister(2), MvNormal(zeros(2), I), 5)))'
    @test rand!(MersenneTwister(2), d, output) === output
    @test output ≈ expected_samples
    buffer = fill(NaN, 2, 6)
    @test rand!(MersenneTwister(3), d, view(buffer, :, 2:4)) === view(buffer, :, 2:4)
    @test all(isfinite, buffer[:, 2:4])
    @test all(isnan, buffer[:, [1, 5, 6]])
    @test mean(d; quad = GaussHermiteWeights(2, L)) ≈ mean(expected)
    @test MonteCarloWeights(4, L).distribution == Normal()
    @test LatinHypercubeWeights(4, L).distribution == Normal()
    @test_throws ErrorException GaussLegendreWeights(2, L)

    signed = TransportMapDistribution(LinearMap([1.0, -2.0], [-2.0, 3.0]))
    @test logpdf(signed, X') ≈ logpdf(expected, X')
    @test cov(signed) ≈ cov(expected)
    large = LinearMap(zeros(2), fill(1.0e200, 2))
    @test logpdf(TransportMapDistribution(large), zeros(2)) ≈ -log(2π) - 400log(10)
    @test entropy(TransportMapDistribution(large)) ≈ log(2π * exp(1)) + 400log(10)
    @test_throws ArgumentError TransportMapDistribution(LinearMap())
    @test_throws ArgumentError TransportMapDistribution(LinearMap([0.0], [0.0]))
    @test_throws ArgumentError TransportMapDistribution(LinearMap([0.0], [Inf]))
    @test_throws DimensionMismatch TransportMapDistribution(LinearMap([0.0], [1.0, 2.0]))
end

@testitem "LinearMap" setup = [LinearMapSetup] begin
    rng = MersenneTwister(1234)

    @testset "Constructor and basic operations" begin
        samples = randn(rng, 100, 3)
        L = LinearMap(samples)

        @test numberdimensions(L) == 3

        x = randn(rng, 3)
        y = evaluate(L, x)
        x_rec = inverse(L, y)
        @test isapprox(x_rec, x; atol = 1.0e-10, rtol = 1.0e-10)

        # Matrix form
        X = randn(rng, 7, 3)
        Y = evaluate(L, X)
        X_rec = inverse(L, Y)
        @test isapprox(X_rec, X; atol = 1.0e-10, rtol = 1.0e-10)

        # Test constructor with explicit μ and σ
        μ_test = [1.0, 2.0, 3.0]
        σ_test = [0.5, 1.5, 2.5]
        L2 = LinearMap(μ_test, σ_test)

        @test L2.μ == μ_test
        @test L2.σ == σ_test
        @test numberdimensions(L2) == 3

        # Verify it works correctly with specified parameters
        x_test = [2.0, 5.0, 8.0]
        y_test = evaluate(L2, x_test)
        x_recovered = inverse(L2, y_test)
        @test isapprox(x_recovered, x_test; atol = 1.0e-10, rtol = 1.0e-10)
        # Jacobian
        jac = jacobian(L)
        @test isapprox(jac, prod(L.σ); atol = 1.0e-10, rtol = 1.0e-10)
    end

    @testset "Identity LinearMap" begin
        L0 = LinearMap()
        @test numberdimensions(L0) == 0

        x = Float64[]
        @test evaluate(L0, x) == x
        @test inverse(L0, x) == x

        X = Array{Float64}(undef, 2, 0)
        @test size(evaluate(L0, X)) == size(X)
        @test size(inverse(L0, X)) == size(X)

        L2 = LinearMap(2)
        @test numberdimensions(L2) == 2
    end

    @testset "Set parameters" begin
        L = LinearMap(2)
        μ_new = [1.0, -1.0]
        σ_new = [2.0, 0.5]
        setparameters!(L, μ_new, σ_new)

        @test L.μ == μ_new
        @test L.σ == σ_new

        x = [3.0, 0.0]
        y = evaluate(L, x)
        x_rec = inverse(L, y)
        @test isapprox(x_rec, x; atol = 1.0e-10, rtol = 1.0e-10)
    end

    @testset "Show" begin
        L = LinearMap()
        @test_nowarn sprint(show, L)
        @test_nowarn sprint(print, L)
        @test_nowarn display(L)
    end
end
