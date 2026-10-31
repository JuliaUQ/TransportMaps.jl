@testsnippet ComposedMapSetup begin
    using TransportMaps
    using Test
    using Random

end

@testitem "ComposedMap distribution" setup = [ComposedMapSetup] begin
    using Distributions, LinearAlgebra, Statistics

    linear = LinearMap([0.5, -1.0], [2.0, 3.0])
    laplace = LaplaceMap([-1.0 0.2; 0.2 0.8; 1.2 -0.4; 0.4 0.1])
    A = [1.5 0.0; 0.3 0.8]
    b = [0.5, -0.2]
    reference = Normal(0.4, 1.7)
    X = [0.0 0.0; 1.0 -1.0; -0.5 0.2]

    for L in (linear, laplace, LinearMap(), LinearMap([0.5, -1.0], [-2.0, 3.0]))
        pm = PolynomialMap(2, 1, reference, IdentityRectifier(), HermiteBasis())
        setcoefficients!(pm[1], [b[1], A[1, 1]])
        setcoefficients!(pm[2], [b[2], A[2, 1], A[2, 2]])
        C = ComposedMap(L, pm)
        @test C isa AbstractTransportMap
        @test numberdimensions(C) == 2
        B = L isa LaplaceMap ? inv(L.chol) :
            (numberdimensions(L) == 0 ? Matrix{Float64}(I, 2, 2) : Matrix(Diagonal(inv.(L.σ))))
        location = L isa LaplaceMap ? L.mode : (numberdimensions(L) == 0 ? zeros(2) : L.μ)
        W = A * B
        shift = b - W * location
        for direction in (:target, :reference)
            TransportMaps.setforwarddirection!(pm, direction)
            d = TransportMapDistribution(C)
            μ = direction == :target ? W * fill(0.4, 2) + shift : W \ (fill(0.4, 2) - shift)
            transform = direction == :target ? W : inv(W)
            expected = MvNormal(μ, 1.7^2 * transform * transform')
            @test length(d) == 2
            @test mean(d) ≈ mean(expected) atol = 1.0e-7
            @test var(d) ≈ var(expected) atol = 1.0e-7
            @test std(d) ≈ std(expected) atol = 1.0e-7
            @test cov(d) ≈ cov(expected) atol = 1.0e-7
            @test cor(d) ≈ cor(expected) atol = 1.0e-7
            @test entropy(d) ≈ entropy(expected) atol = 1.0e-7
            @test logpullback(C, X) ≈ logpdf(expected, X') atol = 1.0e-10
            @test pdf(d, X') ≈ pdf(expected, X') atol = 1.0e-10
            @test logpdf(d, X') ≈ logpdf(expected, X') atol = 1.0e-10
            @test insupport(d, X') == trues(3)
            @test logabsdetjacobian(C, X) ≈ fill(logabsdet(W)[1], 3)
            @test jacobian(C, X) ≈ fill(det(W), 3)
            @test mean(d; quad = GaussHermiteWeights(2, C)) ≈ mean(expected) atol = 1.0e-7
            @test MonteCarloWeights(4, C).distribution == reference
            @test LatinHypercubeWeights(4, C).distribution == reference
            @test isempty(logpdf(d, zeros(2, 0)))
            @test !insupport(d, [Inf, 0.0])
            @test logpdf(d, [Inf, 0.0]) == -Inf
            @test_throws DimensionMismatch logpdf(d, zeros(3))

            Z = rand(MersenneTwister(7), product_distribution(fill(reference, 2)), 5)
            expected_samples = direction == :target ? evaluate(C, Z') : inverse(C, Z')
            output = fill(NaN, 2, 5)
            @test rand!(MersenneTwister(7), d, output) === output
            @test output ≈ expected_samples' atol = 1.0e-8
            @test all(insupport(d, output))
            point = fill(NaN, 2)
            @test rand!(MersenneTwister(8), d, point) === point
            z = rand(MersenneTwister(8), product_distribution(fill(reference, 2)))
            @test point ≈ (direction == :target ? evaluate(C, z) : inverse(C, z)) atol = 1.0e-8
        end
    end
end

@testitem "ComposedMap bounded endpoint reference" setup = [ComposedMapSetup] begin
    using Distributions, LinearAlgebra, Statistics

    linear = LinearMap([0.5, -1.0], [2.0, 3.0])
    laplace = LaplaceMap([-1.0 0.2; 0.2 0.8; 1.2 -0.4; 0.4 0.1])
    for L in (linear, laplace, LinearMap(), LinearMap([0.5, -1.0], [-2.0, 3.0]))
        # M(y) = y, so the reference endpoints belong to the whole composition,
        # rather than the intermediate standardized coordinates L(z).
        pm = PolynomialMap(2, 1, Uniform(0.0, 1.0), IdentityRectifier())
        setcoefficients!(pm[1], [0.5, 0.5])
        setcoefficients!(pm[2], [0.5, 0.0, 0.5])
        C = ComposedMap(L, pm)
        for direction in (:target, :reference)
            TransportMaps.setforwarddirection!(pm, direction)
            d = TransportMapDistribution(C)
            transform(z) = direction == :target ? evaluate(C, z) : inverse(C, z)
            logdet = logabsdetjacobian(C, zeros(2))
            expected_logdensity = direction == :target ? -logdet : logdet
            expected_entropy = -expected_logdensity
            @test entropy(d) ≈ expected_entropy atol = 1.0e-8
            @test mean(d; quad = GaussLegendreWeights(2, C)) ≈ transform(fill(0.5, 2)) atol = 1.0e-7
            @test MonteCarloWeights(4, C).distribution == Uniform(0.0, 1.0)
            @test LatinHypercubeWeights(4, C).distribution == Uniform(0.0, 1.0)
            for z in ([0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5])
                x = transform(z)
                @test insupport(d, x)
                @test logpdf(d, x) ≈ expected_logdensity atol = 1.0e-8
            end
            for z in ([1.1, 0.5], [0.5, -0.1])
                x = transform(z)
                @test !insupport(d, x)
                @test logpdf(d, x) == -Inf
                @test pdf(d, x) == 0.0
            end
            @test all(insupport(d, rand(MersenneTwister(9), d, 8)))
        end
    end
end

@testitem "ComposedMap nonlinear support and Jacobian" setup = [ComposedMapSetup] begin
    using Distributions, LinearAlgebra, Statistics

    laplace = LaplaceMap([-1.0 0.2; 0.2 0.8; 1.2 -0.4; 0.4 0.1])
    for L in (
            LinearMap([0.5, -1.0], [2.0, 3.0]), laplace,
            LinearMap([0.5, -1.0], [-2.0, 3.0]),
        )
        # Polynomial shear M(y) = (y₁, y₂ + y₁²) has determinant one.
        pm = PolynomialMap(2, 2, Uniform(-1.0, 1.0), IdentityRectifier())
        setcoefficients!(pm[1], [0.0, 1.0, 0.0])
        terms = [basis.multiindexset for basis in pm[2].basisfunctions]
        coefficients = zeros(length(terms))
        coefficients[findfirst(==([0, 0]), terms)] = 1 / 3
        coefficients[findfirst(==([2, 0]), terms)] = 2 / 3
        coefficients[findfirst(==([0, 1]), terms)] = 1.0
        setcoefficients!(pm[2], coefficients)
        C = ComposedMap(L, pm)
        input = [0.2, -0.3]
        h = 1.0e-5
        J = hcat(
            [
                (
                    evaluate(C, input + h * I(2)[:, j]) -
                        evaluate(C, input - h * I(2)[:, j])
                ) / (2h) for j in 1:2
            ]...
        )
        @test jacobian(C, input) ≈ det(J) rtol = 1.0e-8
        @test logabsdetjacobian(C, input) ≈ log(abs(det(J))) rtol = 1.0e-8

        for direction in (:target, :reference)
            TransportMaps.setforwarddirection!(pm, direction)
            d = TransportMapDistribution(C)
            transform(z) = direction == :target ? evaluate(C, z) : inverse(C, z)
            logdet = logabsdetjacobian(C, zeros(2))
            expected = -2log(2) + (direction == :target ? -logdet : logdet)
            for z in ([-1.0, -1.0], [1.0, 1.0], [0.5, -0.5], [-0.2, 0.3])
                x = transform(z)
                @test insupport(d, x)
                @test logpdf(d, x) ≈ expected atol = 1.0e-8
            end
            for z in ([1.1, 0.0], [0.0, -1.1])
                x = transform(z)
                @test !insupport(d, x)
                @test logpdf(d, x) == -Inf
            end
            @test entropy(d) ≈ -expected atol = 1.0e-8
            @test cov(d) ≈ cov(d; quad = GaussLegendreWeights(3, C)) atol = 1.0e-7
            @test var(d) ≈ diag(cov(d)) atol = 1.0e-10
        end
    end
end

@testitem "ComposedMap" setup = [ComposedMapSetup] begin
    rng = MersenneTwister(42)

    # Build a simple PolynomialMap that acts approximately as identity
    pm = PolynomialMap(2, 1, :normal, IdentityRectifier())
    # For degree 1 with our basis choices, setcoefficients to produce identity-like mapping
    # Component 1 (1D linear): coefficients [0, 1] -> f(x)=x
    setcoefficients!(pm.components[1], [0.0, 1.0])
    # Component 2 (2D linear): choose coefficients so M^2(z1,z2) = z2
    # Number of coefficients varies; set a vector where linear term for z2 is 1 and others 0
    coeffs2 = zeros(length(pm.components[2].coefficients))
    # Attempt to place 1.0 at the last coefficient which typically corresponds to z2 linear term
    coeffs2[end] = 1.0
    setcoefficients!(pm.components[2], coeffs2)

    TransportMaps.setforwarddirection!(pm, :reference)

    # Build LinearMap from simple samples (shift-scale)
    samples = randn(rng, 50, 2) .* [2.0 1.0] .+ [0.5 - 0.3]
    L = LinearMap(samples)

    C = ComposedMap(L, pm)

    @test numberdimensions(C) == 2

    # Test evaluate/inverse round-trip
    x = [0.1, -0.2]
    y = evaluate(C, x)
    x_rec = inverse(C, y)
    @test isapprox(x_rec, x; atol = 1.0e-8)

    # Matrix forms
    X = randn(rng, 5, 2)
    Y = evaluate(C, X)
    X_rec = inverse(C, Y)
    @test isapprox(X_rec, X; atol = 1.0e-8)

    # Test pullback scaling: pullback(Composed) == pullback(Polynomial) / prod(σ)
    # Compute pullbacks at a point (use X row)
    x0 = X[1, :]
    pb_C = pullback(C, x0)
    pb_pm = pullback(pm, evaluate(L, x0))
    scale = prod(L.σ)
    @test isapprox(pb_C * scale, pb_pm; atol = 1.0e-8, rtol = 1.0e-8)

    # Test pullback scaling: pullback(Composed) == pullback(Polynomial) / prod(σ) for matrix input
    pb_C = pullback(C, X)
    pb_pm = pullback(pm, evaluate(L, X))
    scale = prod(L.σ)
    @test isapprox(pb_C * scale, pb_pm; atol = 1.0e-8, rtol = 1.0e-8)

    @testset "Show" begin
        @test_nowarn sprint(show, C)
        @test_nowarn sprint(print, C)
        @test_nowarn display(C)
    end
end
