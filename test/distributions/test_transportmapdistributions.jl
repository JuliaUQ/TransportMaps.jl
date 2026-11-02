@testsnippet TransportMapDistributionSetup begin
    using TransportMaps
    using Test
    using Distributions
    using LinearAlgebra
    using Random
    using Statistics
    using Optim
end

@testitem "LinearMap distribution" setup = [TransportMapDistributionSetup] begin

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
    @test pdf(d, permutedims(X)) ≈ pdf(expected, permutedims(X))
    @test loglikelihood(d, permutedims(X)) ≈ loglikelihood(expected, permutedims(X))
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

@testitem "LinearMap exact statistics" setup = [TransportMapDistributionSetup] begin

    L = LinearMap([1.0e16, -2.0], [-2.0, 3.0])
    d = TransportMapDistribution(L)
    @test mean(d) == L.μ
    @test var(d) == [4.0, 9.0]
    @test std(d) == [2.0, 3.0]
    @test cov(d) == [4.0 0.0; 0.0 9.0]
    @test cor(d) == Matrix{Float64}(I, 2, 2)
    @test entropy(d) ≈ 2entropy(Normal()) + log(6)

    # Returned locations cannot mutate the fitted map; later parameter changes
    # must still be reflected in the wrapper.
    location = mean(d)
    location[1] = 0.0
    @test L.μ[1] == 1.0e16
    L.μ[1] = 1.0
    @test mean(d) == [1.0, -2.0]
    quad = GaussHermiteWeights(2, L)
    for statistic in (mean, var, std, cov, cor, entropy)
        @test statistic(d; quad = quad) ≈ statistic(d)
    end
    biased_rule = MonteCarloWeights([2.0 -1.0], [1.0])
    for statistic in (mean, var, std, cov, cor, entropy)
        @test statistic(d; quad = biased_rule) == statistic(d)
    end

    # Exact standard deviations and correlations do not square extreme scales.
    extreme = TransportMapDistribution(LinearMap(zeros(2), [-1.0e200, 1.0e-200]))
    @test std(extreme) == [1.0e200, 1.0e-200]
    @test cor(extreme) == Matrix{Float64}(I, 2, 2)
    @test entropy(extreme) ≈ 2entropy(Normal())

    highdim = TransportMapDistribution(LinearMap(64))
    @test mean(highdim) == zeros(64)
    @test var(highdim) == ones(64)
    @test entropy(highdim) ≈ 64entropy(Normal())
end

@testitem "LaplaceMap distribution" setup = [TransportMapDistributionSetup] begin
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
    @test insupport(d, X') == trues(3)
    @test !insupport(d, [NaN, 0.0])
    @test logpdf(d, [NaN, 0.0]) == -Inf
    @test mean(d; quad = SparseSmolyakWeights(2, L)) ≈ mean(expected)
    point = fill(NaN, 2)
    @test rand!(MersenneTwister(5), d, point) === point
    @test point ≈ inverse(L, rand(MersenneTwister(5), MvNormal(zeros(2), I)))
    output = fill(NaN, 2, 5)
    @test rand!(MersenneTwister(6), d, output) === output
    @test output ≈ inverse(L, permutedims(rand(MersenneTwister(6), MvNormal(zeros(2), I), 5)))'
    @test isempty(logpdf(d, zeros(2, 0)))
    @test_throws DimensionMismatch logpdf(d, zeros(3))
end

@testitem "Polynomial distribution quadrature" setup = [TransportMapDistributionSetup] begin
    map = PolynomialMap(1, 1, Normal(), IdentityRectifier(), HermiteBasis())
    setcoefficients!(map, [1.0, 2.0])
    d = TransportMapDistribution(map)
    quad = MonteCarloWeights(reshape([-1.0, 0.0, 2.0], :, 1), [0.2, 0.3, 0.5])
    reference_mean, reference_variance = 0.8, 1.56
    for direction in (:target, :reference)
        TransportMaps.setforwarddirection!(map, direction)
        shift, scale = direction == :target ? (1.0, 2.0) : (-0.5, 0.5)
        expected_mean = shift + scale * reference_mean
        expected_variance = scale^2 * reference_variance
        @test mean(d; quad = quad) ≈ [expected_mean]
        @test var(d; quad = quad) ≈ [expected_variance]
        @test std(d; quad = quad) ≈ [sqrt(expected_variance)]
        @test cov(d; quad = quad) ≈ fill(expected_variance, 1, 1)
        @test cor(d; quad = quad) ≈ ones(1, 1)
        @test entropy(d; quad = quad) ≈ entropy(Normal()) + log(scale)
        @test mean(d) ≈ [shift]
        @test var(d) ≈ [scale^2]
    end
end

@testitem "LaplaceMap exact statistics" setup = [TransportMapDistributionSetup] begin
    L = LaplaceMap([-1.0 0.2; 0.2 0.8; 1.2 -0.4; 0.4 0.1])
    d = TransportMapDistribution(L)
    expected = MvNormal(L)
    for statistic in (mean, var, std, cov, cor, entropy)
        @test statistic(d) ≈ statistic(expected)
        @test statistic(d; quad = GaussHermiteWeights(2, L)) ≈ statistic(expected)
    end
    location = mean(d)
    location[1] = 100.0
    @test L.mode[1] != 100.0
    L.mode[1] = 1.0e16
    @test mean(d) == L.mode
    @test var(d) ≈ var(expected)
    biased_rule = MonteCarloWeights([2.0 -1.0], [1.0])
    for statistic in (mean, var, std, cov, cor, entropy)
        @test statistic(d; quad = biased_rule) == statistic(d)
    end

    expected_std, expected_cor, expected_entropy = std(d), cor(d), entropy(d)
    L.chol .*= 1.0e200
    @test std(d) ≈ 1.0e200 .* expected_std
    @test cor(d) ≈ expected_cor
    @test entropy(d) ≈ expected_entropy + 400log(10)

    highdim_map = LaplaceMap(randn(MersenneTwister(10), 80, 32))
    highdim = TransportMapDistribution(highdim_map)
    @test mean(highdim) == highdim_map.mode
    @test cov(highdim) ≈ cov(highdim_map)
    @test entropy(highdim) ≈ entropy(MvNormal(highdim_map))
end

@testitem "ComposedMap distribution" setup = [TransportMapDistributionSetup] begin

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
            @test pdf(d, X') ≈ pdf(expected, X') atol = 1.0e-10
            @test logpdf(d, X') ≈ logpdf(expected, X') atol = 1.0e-10
            @test insupport(d, X') == trues(3)
            @test mean(d; quad = GaussHermiteWeights(2, C)) ≈ mean(expected) atol = 1.0e-7
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

@testitem "ComposedMap bounded endpoint reference" setup = [TransportMapDistributionSetup] begin

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

@testitem "ComposedMap nonlinear distribution support" setup = [TransportMapDistributionSetup] begin

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

@testitem "Distributions interface uses column batches" setup = [TransportMapDistributionSetup] begin
    pm = PolynomialMap(2, 1, Normal(0.4, 1.7), IdentityRectifier(), HermiteBasis())
    setcoefficients!(pm[1], [1.0, 1.5])
    setcoefficients!(pm[2], [-0.5, 0.3, 0.8])
    X = evaluate(pm, [0.0 0.0; 0.3 -0.7; 1.2 0.5])
    @testset "Distributions interface uses column batches" begin
        for direction in (:target, :reference)
            TransportMaps.setforwarddirection!(pm, direction)
            distribution = TransportMapDistribution(pm)
            @test logpdf(distribution, X[1, :]) ≈ logpullback(pm, X[1, :])
            @test logpdf(distribution, permutedims(X)) ≈ logpullback(pm, X)
            @test pdf(distribution, permutedims(X)) ≈ pullback(pm, X)
            @test loglikelihood(distribution, permutedims(X)) ≈ sum(logpullback(pm, X))
            @test_throws DimensionMismatch logpdf(distribution, [0.0])
            @test_throws DimensionMismatch logpdf(distribution, X)
        end
    end
end

@testitem "Explicit indices retain reference metadata" setup = [TransportMapDistributionSetup] begin
    indices = [[[0], [1]]]
    configurations = (
        (Uniform(-1.0, 1.0), LegendreBasis(), [0.0, 1.0]),
        (Uniform(0.0, 1.0), ShiftedLegendreBasis(), [0.5, 0.5]),
        (Normal(2.0, 3.0), HermiteBasis(), [0.0, 1.0]),
    )
    for (reference, basis, coefficients) in configurations
        map = PolynomialMap(indices, IdentityRectifier(), basis, reference)
        setcoefficients!(map, coefficients)
        @test map.reference.densitytype == reference
        @test map.forwarddirection == :target
        distribution = TransportMapDistribution(map)
        @test mean(distribution) ≈ [mean(reference)] atol = 1.0e-10
        @test var(distribution) ≈ [var(reference)] atol = 1.0e-10
        @test entropy(distribution) ≈ entropy(reference) atol = 1.0e-10
        for x in (-2.0, 0.25, 0.75, 2.0)
            @test logpdf(distribution, [x]) ≈ logpdf(reference, x) atol = 1.0e-10
        end
        generated = rand(MersenneTwister(919), distribution, 10)
        @test all(insupport(distribution, generated))
        @test all(value -> insupport(reference, value), generated)
    end
end

@testitem "Distribution log-density remains finite in the tails" setup = [TransportMapDistributionSetup] begin
    pm = DiagonalMap(1, 1, Normal(), IdentityRectifier(), HermiteBasis())
    setcoefficients!(pm, [0.0, 1.0])
    for direction in (:target, :reference)
        TransportMaps.setforwarddirection!(pm, direction)
        @test logpdf(TransportMapDistribution(pm), [40.0]) ≈ logpdf(Normal(), 40.0) atol = 1.0e-9
    end
end

@testitem "Transport-map support" setup = [TransportMapDistributionSetup] begin
    @testset "Shifted uniform support" begin
        for direction in (:target, :reference)
            pm = PolynomialMap(1, 1, Uniform(0.0, 1.0), IdentityRectifier(), ShiftedLegendreBasis())
            coefficients = direction == :target ? [11.0, 1.0] : [-4.75, 0.25]
            setcoefficients!(pm, coefficients)
            TransportMaps.setforwarddirection!(pm, direction)
            distribution = TransportMapDistribution(pm)

            for x in (10.0, 10.5, 11.0, 12.0)
                @test insupport(distribution, [x])
                @test logpdf(distribution, [x]) ≈ -log(2) atol = 1.0e-10
                @test logpullback(pm, [x]) ≈ -log(2) atol = 1.0e-10
            end
            for x in (0.0, 1.0, 9.0, 10.0 - 1.0e-10, 12.0 + 1.0e-10, 13.0, NaN, Inf, -Inf)
                @test !insupport(distribution, [x])
                @test logpdf(distribution, [x]) == -Inf
                @test pdf(distribution, [x]) == 0.0
                @test logpullback(pm, [x]) == -Inf
            end
            @test !insupport(distribution, Float64[])
            @test !insupport(distribution, [11.0, 11.0])
            @test insupport(distribution, [11])
            @test insupport(distribution, view([0.0, 11.0, 0.0], 2:2))
            @test insupport(distribution, [9.0 10.0 11.0 12.0 13.0]) ==
                [false, true, true, true, false]
            @test_throws DimensionMismatch insupport(distribution, zeros(2, 3))
        end
    end

    @testset "Triangular support is not a bounding box" begin
        for direction in (:target, :reference)
            pm = PolynomialMap(2, 1, Uniform(-1.0, 1.0), IdentityRectifier(), LegendreBasis())
            sign = direction == :target ? 1.0 : -1.0
            setcoefficients!(pm[1], [0.0, 1.0])
            setcoefficients!(pm[2], [0.0, sign, 1.0])
            TransportMaps.setforwarddirection!(pm, direction)
            distribution = TransportMapDistribution(pm)

            for x in ([0.0, 0.0], [0.5, 1.25], [-0.5, -1.25], [1.0, 2.0], [-1.0, -2.0])
                @test insupport(distribution, x)
            end
            for x in ([0.9, -0.9], [-0.9, 0.9], [0.0, 1.5], [1.1, 1.1])
                @test !insupport(distribution, x)
                @test logpdf(distribution, x) == -Inf
                @test pullback(pm, x) == 0.0
            end
        end
    end

    @testset "Curved conditional boundaries" begin
        for direction in (:target, :reference)
            pm = PolynomialMap(2, 2, Uniform(-1.0, 1.0), IdentityRectifier(), LegendreBasis())
            sign = direction == :target ? 1.0 : -1.0
            setcoefficients!(pm[1], [0.0, 1.0, 0.0])
            terms = [basis.multiindexset for basis in pm[2].basisfunctions]
            coefficients = zeros(length(terms))
            coefficients[findfirst(==([0, 0]), terms)] = sign / 3
            coefficients[findfirst(==([2, 0]), terms)] = 2 * sign / 3
            coefficients[findfirst(==([0, 1]), terms)] = 1.0
            setcoefficients!(pm[2], coefficients)
            TransportMaps.setforwarddirection!(pm, direction)
            distribution = TransportMapDistribution(pm)

            for x in ([0.0, 0.0], [0.5, 1.2], [0.5, -0.5], [1.0, 2.0], [-1.0, 0.0])
                @test insupport(distribution, x)
                @test logpdf(distribution, x) ≈ -2 * log(2) atol = 1.0e-10
            end
            for x in ([0.0, 1.5], [0.5, -0.9], [1.1, 1.0])
                @test !insupport(distribution, x)
                @test logpdf(distribution, x) == -Inf
                @test pullback(pm, x) == 0.0
            end
        end
    end

    @testset "Unbounded reference" begin
        pm = PolynomialMap(1, 1, Normal(), IdentityRectifier(), HermiteBasis())
        setcoefficients!(pm, [10.0, 2.0])
        distribution = TransportMapDistribution(pm)
        for direction in (:target, :reference)
            TransportMaps.setforwarddirection!(pm, direction)
            @test insupport(distribution, [-40.0])
            @test insupport(distribution, [40.0])
            @test insupport(distribution, [1.0e4])
            @test !insupport(distribution, [NaN])
            @test !insupport(distribution, [Inf])
            @test !insupport(distribution, [-Inf])
        end

        # This map has derivative exp(-z²), hence a bounded image despite its
        # normal reference. Points outside the image must terminate and fail.
        bounded_map = PolynomialMap(1, 3, Normal(), ExpRectifier(), HermiteBasis())
        setcoefficients!(bounded_map, [0.0, -1.0, 0.0, -1 / 3])
        bounded_distribution = TransportMapDistribution(bounded_map)
        @test insupport(bounded_distribution, [0.0])
        @test insupport(bounded_distribution, [0.25])
        @test !insupport(bounded_distribution, [10.0])
        @test !insupport(bounded_distribution, [-10.0])
        @test logpdf(bounded_distribution, [0.0]) ≈ logpdf(Normal(), 0.0)
        for x in ([10.0], [-10.0])
            @test logpdf(bounded_distribution, x) == -Inf
            @test pdf(bounded_distribution, x) == 0.0
            @test logpullback(bounded_map, x) == -Inf
            @test pullback(bounded_map, x) == 0.0
        end
        X = reshape([-10.0, 0.0, 10.0], :, 1)
        expected = [-Inf, logpdf(Normal(), 0.0), -Inf]
        @test logpullback(bounded_map, X) ≈ expected
        @test logpdf(bounded_distribution, permutedims(X)) ≈ expected


    end
end

@testitem "Shared inversion and centered variance" setup = [TransportMapDistributionSetup] begin

    struct InverseCountingRectifier <: AbstractRectifierFunction
        calls::Base.RefValue{Int}
    end
    function (rectifier::InverseCountingRectifier)(x::Real)
        rectifier.calls[] += 1
        return x
    end

    @testset "Log-density solves the inverse once" begin
        calls = Ref(0)
        pm = PolynomialMap(1, 1, Normal(), InverseCountingRectifier(calls), HermiteBasis())
        setcoefficients!(pm, [1.0, 2.0])
        distribution = TransportMapDistribution(pm)
        @test insupport(distribution, [2.5])
        support_calls = calls[]
        calls[] = 0
        result = logpdf(distribution, [2.5])
        # Density evaluation adds one diagonal-Jacobian evaluation to the same
        # inverse solve performed by insupport, without a second root search.
        @test calls[] == support_calls + 1
        @test result ≈ logpdf(Normal(1.0, 2.0), 2.5) atol = 1.0e-12
    end

    @testset "Variance is stable under large translations" begin
        scales = [2.0, 3.0]
        for reference in (Normal(0.4, 1.7), Uniform(-1.0, 1.0))
            basis = reference isa Normal ? HermiteBasis() : LegendreBasis()
            for offset in (1.0e8, 1.0e10), direction in (:target, :reference)
                pm = DiagonalMap(2, 1, reference, IdentityRectifier(), basis)
                setcoefficients!(pm[1], [offset, scales[1]])
                setcoefficients!(pm[2], [-offset, scales[2]])
                TransportMaps.setforwarddirection!(pm, direction)
                distribution = TransportMapDistribution(pm)
                expected = direction == :target ?
                    var(reference) .* scales .^ 2 : var(reference) ./ scales .^ 2
                tensor_quad = TensorProductWeights(2, pm)
                for quad in (SparseSmolyakWeights(3, pm), tensor_quad)
                    variances = var(distribution; quad = quad)
                    @test variances ≈ expected rtol = 1.0e-5 atol = 1.0e-6
                    @test variances ≈ diag(cov(distribution; quad = quad)) atol = 1.0e-12
                    @test std(distribution; quad = quad) ≈ sqrt.(expected) rtol = 1.0e-5 atol = 1.0e-6
                end
                @test var(distribution) ≈ expected rtol = 1.0e-5 atol = 1.0e-6
            end
        end
    end
end

@testitem "BOD posterior density inversion" setup = [TransportMapDistributionSetup] begin
    using Logging

    times = [1, 2, 3, 4, 5]
    observations = [0.18, 0.32, 0.42, 0.49, 0.54]
    noise = sqrt(1.0e-3)
    function logposterior(θ)
        A = 0.4 + (1.2 - 0.4) * cdf(Normal(), θ[1])
        B = 0.01 + (0.31 - 0.01) * cdf(Normal(), θ[2])
        likelihood = sum(
            logpdf(Normal(A * (1 - exp(-B * t)), noise), value)
                for (t, value) in zip(times, observations)
        )
        return likelihood + logpdf(Normal(), θ[1]) + logpdf(Normal(), θ[2])
    end
    map = PolynomialMap(2, 3, Normal(), Softplus(), LinearizedHermiteBasis())
    optimize!(map, MapTargetDensity(logposterior), GaussHermiteWeights(3, map))
    posterior = TransportMapDistribution(map)

    # This is the plotting grid from the BOD documentation example. Previously
    # several inversions oscillated at round-off accuracy for 10,000 iterations.
    values = @test_logs min_level = Logging.Warn [
        pdf(posterior, [θ₁, θ₂])
            for θ₁ in range(-0.5, 1.5; length = 100), θ₂ in range(-0.5, 3.0; length = 100)
    ]
    @test all(isfinite, values)
    @test all(>=(0.0), values)
    @test maximum(values) > 0.0

    for point in ([-0.4393939393939394, 0.7727272727272727], [-0.03535353535353535, 0.13636363636363635])
        reference_point = TransportMaps._reference_point_in_support(map, point)
        @test !isnothing(reference_point)
        @test evaluate(map, reference_point) ≈ point atol = 1.0e-13
    end
end

@testitem "Density refitting updates distribution direction" setup = [TransportMapDistributionSetup] begin
    samples = reshape(collect(range(-2.0, 2.0; length = 101)), :, 1)
    expected = Normal(2.0, 3.0)
    target = MapTargetDensity(x -> logpdf(expected, x[1]))
    for use_precomputed in (false, true)
        map = PolynomialMap(1, 1)
        sample_result = optimize!(map, samples)
        @test Optim.converged(sample_result.optimization_results[1])
        @test map.forwarddirection == :reference
        # Retain the wrapper across refitting: it must follow the new direction.
        distribution = TransportMapDistribution(map)
        quadrature = GaussHermiteWeights(3, map)
        fitting_rule = use_precomputed ?
            PrecomputedMapBasis(map, quadrature.points, quadrature.weights) : quadrature
        result = optimize!(map, target, fitting_rule)
        @test Optim.converged(result)
        @test map.forwarddirection == :target
        @test mean(distribution) ≈ [mean(expected)] atol = 1.0e-5
        @test var(distribution) ≈ [var(expected)] atol = 1.0e-5
        @test entropy(distribution) ≈ entropy(expected) atol = 1.0e-5
        for x in (-1.0, 2.0, 5.0)
            @test logpdf(distribution, [x]) ≈ logpdf(expected, x) atol = 1.0e-5
        end
        generated = rand(MersenneTwister(918), distribution, 5000)
        @test size(generated) == (1, 5000)
        @test mean(generated) ≈ mean(expected) atol = 0.15
        @test std(generated) ≈ std(expected) atol = 0.15
    end
end
