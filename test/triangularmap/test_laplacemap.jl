@testsnippet LaplaceMapSetup begin
    using TransportMaps
    using Test
    using LinearAlgebra
    using Random
    using Optim
    using Distributions
    using Logging
    import DifferentiationInterface: AutoFiniteDiff

end

@testitem "LaplaceMap distribution" setup = [LaplaceMapSetup] begin
    using Statistics
    samples = [-1.0 0.2; 0.2 0.8; 1.2 -0.4; 0.4 0.1]
    L = LaplaceMap(samples)
    d = TransportMapDistribution(L)
    expected = MvNormal(L)
    X = [0.0 0.0; 1.0 -2.0; -0.5 0.3]
    @test mean(d) ≈ mean(expected)
    @test var(d) ≈ var(expected)
    @test std(d) ≈ std(expected)
    @test cov(d) ≈ cov(expected)
    @test cor(d) ≈ cor(expected)
    @test entropy(d) ≈ entropy(expected)
    @test logpdf(d, X') ≈ logpdf(expected, X')
    @test pullback(L, X) ≈ pdf(expected, X')
    @test jacobian(L, X) ≈ fill(inv(jacobian(L)), 3)
    @test logabsdetjacobian(L, X) ≈ fill(-log(jacobian(L)), 3)
    @test insupport(d, X') == trues(3)
    @test !insupport(d, [NaN, 0.0])
    @test logpdf(d, [NaN, 0.0]) == -Inf
    @test mean(d; quad = SparseSmolyakWeights(2, L)) ≈ mean(expected)
    @test MonteCarloWeights(4, L).distribution == Normal()
    @test LatinHypercubeWeights(4, L).distribution == Normal()
    point = fill(NaN, 2)
    @test rand!(MersenneTwister(5), d, point) === point
    @test point ≈ inverse(L, rand(MersenneTwister(5), MvNormal(zeros(2), I)))
    output = fill(NaN, 2, 5)
    @test rand!(MersenneTwister(6), d, output) === output
    @test output ≈ inverse(L, permutedims(rand(MersenneTwister(6), MvNormal(zeros(2), I), 5)))'
    @test isempty(logpdf(d, zeros(2, 0)))
    @test_throws DimensionMismatch logpdf(d, zeros(3))
end

@testitem "LaplaceMap from Samples" setup = [LaplaceMapSetup] begin
    # Create samples from a Laplace distribution
    rng = MersenneTwister(123)
    n_samples = 10
    samples = randn(rng, n_samples, 2)

    # Build LaplaceMap from samples
    L_map = LaplaceMap(samples)

    @test numberdimensions(L_map) == 2

    # Test evaluate/inverse round-trip
    x = [0.5, 0.2]
    y = evaluate(L_map, x)
    x_rec = inverse(L_map, y)
    @test isapprox(x_rec, x; atol = 1.0e-8)

    # Matrix forms
    X = randn(rng, 5, 2)
    Y = evaluate(L_map, X)
    X_rec = inverse(L_map, Y)
    @test isapprox(X_rec, X; atol = 1.0e-8)

    # Jacobian
    jac = jacobian(L_map)
    @test isapprox(jac, abs(det(L_map.chol)); atol = 1.0e-10)

    # Test mean and covariance
    μ = mean(samples, dims = 1) |> vec
    @test isapprox(mean(L_map), μ; atol = 1.0e-8)
    @test isapprox(cov(L_map), cov(samples); atol = 1.0e-8)
    @test isapprox(mode(L_map), μ; atol = 1.0e-8)
    @test isapprox(cov(L_map), L_map.chol * L_map.chol'; atol = 1.0e-8)

    @testset "Show" begin
        @test_nowarn sprint(show, L_map)
        @test_nowarn sprint(print, L_map)
        @test_nowarn display(L_map)
    end
end


@testitem "LaplaceMap from Density" setup = [LaplaceMapSetup] begin

    density(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1])
    target = MapTargetDensity(density)

    x0 = [0.5, 1.0]
    L_map = LaplaceMap(target, x0)

    @test numberdimensions(L_map) == 2
    @test mean(L_map) ≈ [0.0, 0.0]
    @test cov(L_map) ≈ [1.0 1.0; 1.0 2.0]

    # Finite Difference approximation of Hessian
    target_fd = MapTargetDensity(density, AutoFiniteDiff())
    L_fd = LaplaceMap(target_fd, x0)

    @test isapprox(mean(L_fd), [0.0, 0.0]; atol = 1.0e-6)
    @test isapprox(cov(L_fd), cov(L_map); atol = 1.0e-6)

    # Finite Difference approximation of Hessian (analytical gradient)
    target_fd2 = MapTargetDensity(density, x -> [x[2] - 2 * x[1], x[1] - x[2]])
    L_fd2 = LaplaceMap(target_fd2, x0)

    @test isapprox(mean(L_fd2), [0.0, 0.0]; atol = 1.0e-6)
    @test isapprox(cov(L_fd2), cov(L_map); atol = 1.0e-6)

    # Error handling
    options = Optim.Options(iterations = 1)
    @test_throws "LaplaceMap optimization did not converge." LaplaceMap(target, x0; options = options)

    # MvNormal
    @test_logs min_level = Logging.Warn MvNormal(L_map)
    @test_logs min_level = Logging.Warn MvNormal(L_fd)

    # Evaluate map
    x_test = [0.1, 0.2]
    @test L_map(x_test) ≈ evaluate(L_map, x_test)

    X_test = randn(5, 2)
    @test L_map(X_test) ≈ evaluate(L_map, X_test)
end
