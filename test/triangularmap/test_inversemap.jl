@testsnippet InverseMapSetup begin
    using TransportMaps
    using Distributions
end

@testsnippet DirectionMapSetup begin
    using TransportMaps

    struct DirectionalTestMap <: AbstractTransportMap end
    TransportMaps.numberdimensions(::DirectionalTestMap) = 1
    TransportMaps.forwarddirection(::DirectionalTestMap) = :target

    struct UnknownDirectionTestMap <: AbstractTransportMap end
    TransportMaps.numberdimensions(::UnknownDirectionTestMap) = 1
end

@testitem "Map forward directions" setup = [InverseMapSetup, DirectionMapSetup] begin
    L = LinearMap([1.0], [2.0])
    laplace = LaplaceMap(reshape([-1.0, 0.0, 2.0], :, 1))
    marginal = MarginalMap([Normal(1.0, 2.0)])
    M = PolynomialMap(1, 1)

    @test forwarddirection(L) === :reference
    @test forwarddirection(laplace) === :reference
    @test forwarddirection(marginal) === :target
    @test forwarddirection(M) === :target
    @test forwarddirection(inv(L)) === :target
    @test forwarddirection(inv(laplace)) === :target
    @test forwarddirection(inv(marginal)) === :reference

    inverse_M = inv(M)
    C = ComposedMap(inverse_M)
    @test forwarddirection(inverse_M) === :reference
    @test forwarddirection(C) === :reference
    M.forwarddirection = :reference
    @test forwarddirection(inverse_M) === :target
    @test forwarddirection(C) === :reference

    custom = DirectionalTestMap()
    @test forwarddirection(inv(custom)) === :reference
    @test forwarddirection(ComposedMap(custom)) === :target
    @test forwarddirection(ComposedMap(inv(custom))) === :reference
    unknown = UnknownDirectionTestMap()
    @test_throws ArgumentError forwarddirection(inv(unknown))
    @test forwarddirection(ComposedMap(inv(unknown))) === :reference
    @test forwarddirection(ComposedMap(InverseMap(inv(unknown)))) === :reference

    M.forwarddirection = :unknown
    @test_throws ArgumentError forwarddirection(inverse_M)
end

@testitem "InverseMap nonlinear Jacobian" setup = [InverseMapSetup] begin
    using LinearAlgebra

    M = PolynomialMap(2, 2)
    setcoefficients!(M.components[1], [1.0, 0.5, 0.2])
    setcoefficients!(M.components[2], [1.0, 0.3, 0.1, 0.05, 0.02, 0.01])
    inverse_map = inv(M)
    X = [0.2 -0.3; 0.8 0.4]
    Y = evaluate(M, X)
    h = 1.0e-5
    expected = map(eachrow(Y)) do y
        J = hcat(
            [
                (
                    evaluate(inverse_map, y + h * I(2)[:, j]) -
                        evaluate(inverse_map, y - h * I(2)[:, j])
                ) / (2h)
                    for j in 1:2
            ]...
        )
        det(J)
    end

    @test jacobian(inverse_map, Y) ≈ expected rtol = 1.0e-6
    @test logabsdetjacobian(inverse_map, Y) ≈ log.(abs.(expected)) rtol = 1.0e-6
    @test jacobian(inverse_map, Y[1, :]) ≈ expected[1] rtol = 1.0e-6
    @test logabsdetjacobian(inverse_map, Y[1, :]) ≈ log(abs(expected[1])) rtol = 1.0e-6
end

@testitem "InverseMap composition direction" setup = [InverseMapSetup] begin
    L = LinearMap([1.0], [2.0])
    C = ComposedMap(inv(L))
    @test C.forwarddirection === :target
    @test logpdf(TransportMapDistribution(C), [1.6]) ≈ logpdf(Normal(1.0, 2.0), 1.6)

    M = PolynomialMap(1, 1, Normal(0.7, 1.3))
    inverse_M = ComposedMap(inv(M))
    @test inverse_M.forwarddirection === :reference
    @test inverse_M.reference === M.reference
    @test ComposedMap(inv(ComposedMap(M))).reference === M.reference
    @test ComposedMap(inv(L), M).forwarddirection === :target
    @test_throws ArgumentError ComposedMap(inv(L), inv(M))
    M.forwarddirection = :reference
    @test ComposedMap(inv(M)).forwarddirection === :target

    @test ComposedMap(C, inv(C)).forwarddirection === :target
    @test ComposedMap(inv(L); forwarddirection = :reference).forwarddirection === :reference
end

@testitem "InverseMapFromDensity" setup = [InverseMapSetup] begin

    M = PolynomialMap(2, 2)
    quadrature = GaussHermiteWeights(3, 2)

    banana_density(x) = logpdf(Normal(), (x[1])) + logpdf(Normal(), x[2] - (x[1])^2)
    target = MapTargetDensity(banana_density)

    L = TransportMaps.LaplaceMap(target, [0.0, 0.0])
    newtarget = MapTargetDensity(x -> banana_density(inverse(L, x)))

    optimize!(M, newtarget, quadrature)

    C = L^-1 ∘ M
    Cinv = C^-1
    CCinv = Cinv^-1

    CtimesCinv = C ∘ Cinv

    samples = permutedims(rand(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), 100))
    pushed_samples = C(samples)

    C_distr = TransportMapDistribution(C)

    @test_throws(ArgumentError, TransportMapDistribution(Cinv))
    @test CCinv === C
    @test Cinv(C(samples)) ≈ samples
    @test CtimesCinv(samples) ≈ samples
    @test all(isapprox(pdf(C_distr, permutedims(pushed_samples)), pullback(Cinv, target, samples); atol = 1.0e-2))
    @test all(isapprox(logpdf(C_distr, permutedims(pushed_samples)), logpullback(Cinv, target, samples); atol = 1.0e0))
end

@testitem "InverseMapFromSamples" setup = [InverseMapSetup] begin

    function generate_banana_samples(number_samples)
        latent_samples = randn(number_samples, 2)
        return hcat(
            latent_samples[:, 1],
            latent_samples[:, 1] .^ 3 .+ latent_samples[:, 2],
        )
    end

    samples = generate_banana_samples(1000)

    M = PolynomialMap(2, 2)
    L = LaplaceMap(samples)

    optimize!(M, L(samples))

    C = M ∘ L
    Cinv = C^-1

    Cdensity = TransportMapDistribution(C)

    mapped = C(samples)
    inv_mapped = inv(C)(mapped)

    @test_throws(ArgumentError, TransportMapDistribution(Cinv))
    @test mean(inv_mapped) .≈ mean(samples) atol = 1.0e-1
    @test all(isapprox(cov(inv_mapped), cov(samples); atol = 1.0e-1))
    @test all(isapprox(pdf(Cdensity, permutedims(samples)), pushforward(Cinv, samples); atol = 1.0e-2))
    @test all(isapprox(logpdf(Cdensity, permutedims(samples)), logpushforward(Cinv, samples); atol = 1.0e-2))
end
