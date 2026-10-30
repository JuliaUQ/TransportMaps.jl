@testitem "PriorMap transformations and derivatives" begin
    using TransportMaps, Distributions, ForwardDiff, LinearAlgebra

    marginals = [Normal(2, 3), Uniform(-2, 4), LogNormal(0.2, 0.7), Beta(2, 3), Gamma(3, 2)]
    P = PriorMap(marginals)
    z = [-0.4, 0.2, 0.7, -0.8, 0.3]
    x = evaluate(P, z)
    @test P.marginals isa Vector{<:ContinuousUnivariateDistribution}
    @test P.marginals == marginals
    @test P.marginals !== marginals
    @test P isa AbstractTransportMap
    @test P.forwarddirection == :target
    @test numberdimensions(P) == 5
    @test P(z) == x
    @test cdf.(marginals, x) ≈ cdf.(Normal(), z)
    @test inverse(P, x) ≈ z atol = 1.0e-12
    Z = [z'; -z'; zeros(5)']
    X = evaluate(P, Z)
    @test inverse(P, X) ≈ Z atol = 1.0e-12
    @test evaluate(P, zeros(0, 5)) == zeros(0, 5)
    @test inverse(P, zeros(0, 5)) == zeros(0, 5)

    J = ForwardDiff.jacobian(u -> evaluate(P, u), z)
    expected = pdf.(Normal(), z) ./ pdf.(marginals, x)
    @test J ≈ Diagonal(expected)
    h = 1.0e-5
    identity_matrix = Matrix{Float64}(I, 5, 5)
    finite_difference = hcat([(evaluate(P, z + h * identity_matrix[:, i]) - evaluate(P, z - h * identity_matrix[:, i])) / (2h) for i in 1:5]...)
    @test J ≈ finite_difference rtol = 1.0e-8
    @test ForwardDiff.jacobian(u -> inverse(P, u), x) * J ≈ I
    @test jacobian(P, z) ≈ det(J)
    @test logabsdetjacobian(P, z) ≈ logdet(J)
    @test inverse_jacobian(P, x) ≈ 1 / det(J)
    @test jacobian(P, Z) ≈ exp.(logabsdetjacobian(P, Z))
    @test inverse_jacobian(P, X) ≈ 1 ./ jacobian(P, Z)
    @test logpdf(P, x) ≈ sum(logpdf.(marginals, x))
    @test logpdf(P, X) ≈ [logpdf(P, row) for row in eachrow(X)]
    @test pullback(P, X) == pdf(P, X)
    @test insupport(P, x)

    # Stable tails, including values where cdf(Normal(), z) rounds to one.
    tails = [-10.0 10.0; 10.0 -10.0]
    tail_map = PriorMap([Normal(2, 3), LogNormal(0.2, 0.7)])
    @test evaluate(tail_map, tails) ≈ [2 .+ 3 .* tails[:, 1] exp.(0.2 .+ 0.7 .* tails[:, 2])]
    @test inverse(tail_map, evaluate(tail_map, tails)) ≈ tails
    @test isfinite(logabsdetjacobian(tail_map, [40.0, 10.0]))
end

@testitem "PriorMap validation and composition" begin
    using TransportMaps, Distributions
    @test_throws ArgumentError PriorMap(ContinuousUnivariateDistribution[])
    @test_throws MethodError PriorMap([Bernoulli(0.5)])
    @test_throws MethodError PriorMap([Normal(), Poisson(2)])
    @test_throws MethodError PriorMap([MvNormal(zeros(2), ones(2))])
    @test_throws MethodError PriorMap((Normal(),))
    P = PriorMap([Uniform(-2, 4), LogNormal(0.2, 0.7)])
    for operation in (evaluate, inverse, jacobian, logabsdetjacobian, logpdf)
        @test_throws DimensionMismatch operation(P, zeros(3))
        @test_throws DimensionMismatch operation(P, zeros(2, 3))
    end
    @test_throws DomainError inverse(P, [-3.0, 1.0])
    @test inverse(P, [-2.0, 1.0])[1] == -Inf
    @test inverse(P, [4.0, 1.0])[1] == Inf
    @test !insupport(P, [-3.0, 1.0])
    @test logpdf(P, [-3.0, 1.0]) == -Inf

    # Affine nonidentity correction makes order and density factors observable.
    T = LinearMap([0.2, -0.1], [0.8, 1.2])
    C = ComposedMap(T, P; forwarddirection = :target)
    z = [0.3, -0.7]
    x = evaluate(C, z)
    @test C.forwarddirection == :target
    @test inverse(C, x) ≈ z
    @test x ≈ evaluate(P, evaluate(T, z))
    @test logpdf(C, x) ≈ logpdf(MapReferenceDensity(), z) - logabsdetjacobian(C, z)
    @test logpdf(C, [-3.0, 1.0]) == -Inf
    @test pdf(C, [0.0, -1.0]) == 0
    @test logpdf(C, [-2.0, 1.0]) == -Inf
    @test logpdf(ComposedMap(ComposedMap(T, P; forwarddirection = :target)), [-3.0, 1.0]) == -Inf
    @test logpdf(ComposedMap(P), x) ≈ logpdf(P, x)
end

@testitem "PriorMap Bayesian density fitting" begin
    using TransportMaps, Distributions, ForwardDiff, Optim
    # A lognormal prior and observation of log(x) have an exact lognormal posterior.
    μ, σ, y, τ = 0.3, 0.7, 0.8, 0.4
    P = PriorMap([LogNormal(μ, σ)])
    reference = MapReferenceDensity()
    loglikelihood(x) = logpdf(Normal(log(x[1]), τ), y)
    target = MapTargetDensity(z -> loglikelihood(evaluate(P, z)) + logpdf(reference, z))
    z = [0.2]
    @test grad_logpdf(target, z) ≈ [(y - μ - σ * z[1]) * σ / τ^2 - z[1]]
    @test logpdf(target, z) ≈ loglikelihood(evaluate(P, z)) + logpdf(P, evaluate(P, z)) + logabsdetjacobian(P, z)
    T = PolynomialMap(1, 1)
    result = optimize!(T, target, GaussHermiteWeights(8, 1); δ = 0.0, options = Optim.Options(iterations = 200, g_tol = 1.0e-9))
    @test Optim.converged(result)
    C = ComposedMap(T, P; forwarddirection = :target)
    posterior_variance = 1 / (1 / σ^2 + 1 / τ^2)
    posterior_mean = posterior_variance * (μ / σ^2 + y / τ^2)
    posterior = LogNormal(posterior_mean, sqrt(posterior_variance))
    for u in (-1.0, 0.0, 1.0)
        x = evaluate(C, [u])
        @test x[1] ≈ exp(posterior_mean + sqrt(posterior_variance) * u) rtol = 1.0e-6
        @test logpdf(C, x) ≈ logpdf(posterior, x[1]) atol = 1.0e-6
    end
end

@testitem "PriorMap configurable reference" begin
    using TransportMaps, Distributions, ForwardDiff, LinearAlgebra

    marginals = [Normal(2, 3), Uniform(-2, 4), LogNormal(0.2, 0.7), Beta(2, 3)]
    probabilities = [0.2, 0.4, 0.7, 0.8]
    for reference in (Uniform(0, 1), Uniform(-1, 1), Normal(2, 3), Beta(2, 4))
        P = PriorMap(marginals; reference)
        @test P isa PriorMap{eltype(marginals), typeof(reference)}
        z = quantile.(reference, probabilities)
        x = evaluate(P, z)
        @test P.reference.densitytype == reference
        @test cdf.([marginals...], x) ≈ probabilities
        @test inverse(P, x) ≈ z atol = 1.0e-11
        Z = [z'; quantile.(reference, 1 .- probabilities)']
        @test inverse(P, evaluate(P, Z)) ≈ Z atol = 1.0e-11
        @test PriorMap([marginals...]; reference).reference.densitytype == reference

        expected = pdf.(reference, z) ./ [pdf(d, x[i]) for (i, d) in enumerate(marginals)]
        J = ForwardDiff.jacobian(u -> evaluate(P, u), z)
        @test J ≈ Diagonal(expected)
        h, directions = 1.0e-5, Matrix{Float64}(I, 4, 4)
        numerical_J = hcat([(evaluate(P, z + h * directions[:, i]) - evaluate(P, z - h * directions[:, i])) / (2h) for i in 1:4]...)
        @test J ≈ numerical_J rtol = 1.0e-8
        @test ForwardDiff.jacobian(u -> inverse(P, u), x) * J ≈ I
        @test logabsdetjacobian(P, z) ≈ sum(log, expected)
        @test inverse_jacobian(P, x) ≈ 1 / prod(expected)

        C = ComposedMap(LinearMap(4), P; forwarddirection = :target)
        @test C.reference === P.reference
        @test logpdf(C, x) ≈ logpdf(P, x)
        @test logpdf(C, x) ≈ logpdf(P.reference, z) - logabsdetjacobian(P, z)
        target = MapTargetDensity(u -> -sum(abs2, evaluate(P, u)) / 2 + logpdf(P.reference, u))
        expected_score = -x .* expected + [Distributions.gradlogpdf(reference, zi) for zi in z]
        @test grad_logpdf(target, z) ≈ expected_score

        if reference isa Uniform
            q = GaussLegendreWeights(4, P)
            @test sum(q.weights) ≈ 1
            @test vec(q.points' * q.weights) ≈ fill(mean(reference), 4) atol = 1.0e-14
            T = PolynomialMap(4, 1, reference)
            @test T.reference.densitytype == P.reference.densitytype
            @test GaussLegendreWeights(4, T).points == q.points
            @test_throws DomainError evaluate(P, fill(minimum(reference) - 0.1, 4))
            @test_throws DomainError logabsdetjacobian(P, fill(maximum(reference) + 0.1, 4))
        end
    end
    @test_throws ArgumentError PriorMap([Normal()]; reference = Bernoulli(0.5))
    @test_throws ArgumentError PriorMap([Normal()]; reference = MvNormal(zeros(2), ones(2)))
    # Uniform-to-uniform maps preserve finite boundaries and exact affine slopes.
    P = PriorMap([Uniform(-2, 4)]; reference = Uniform(0, 1))
    @test evaluate(P, [0.0]) == [-2.0]
    @test evaluate(P, [1.0]) == [4.0]
    @test inverse(P, [-2.0]) == [0.0]
    @test inverse(P, [4.0]) == [1.0]
    @test logabsdetjacobian(P, [1.0]) ≈ log(6)
    @test logpdf(ComposedMap(P), [4.0]) ≈ -log(6)
    # Avoid reconstructing a rounded tail probability when differentiating the inverse.
    normal_prior = PriorMap([Normal()]; reference = Uniform(0, 1))
    @test inverse_jacobian(normal_prior, [10.0]) ≈ pdf(Normal(), 10.0)
    @test logpdf(ComposedMap(normal_prior), [10.0]) ≈ logpdf(Normal(), 10.0)
end
