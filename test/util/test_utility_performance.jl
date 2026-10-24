@testitem "Density batch buffers preserve callbacks and AD" begin
    using Distributions: Normal, Uniform
    using DifferentiationInterface: AutoForwardDiff, AutoFiniteDiff
    using Random

    rng = MersenneTwister(941)
    storage = 0.3 .* randn(rng, 66, 5)
    points = view(storage, 1:2:66, 2:5)
    original = copy(storage)
    # This signature intentionally requires a Vector, including for AD inputs.
    logπ(x::Vector{<:Real}) = -sum(abs2, x) / 2
    analytic(x::Vector{Float64}) = -x

    for threaded in (false, true)
        targets = [
            MapTargetDensity(logπ, analytic; threaded),
            MapTargetDensity(logπ, AutoForwardDiff(); threaded),
            MapTargetDensity(logπ, AutoForwardDiff(), 4; threaded),
            MapTargetDensity(logπ, AutoFiniteDiff(), 4; threaded),
        ]
        for target in targets
            for _ in 1:3
                @test grad_logpdf(target, points) ≈ -points rtol = 1.0e-6 atol = 1.0e-8
                @test storage == original
            end
            @test size(grad_logpdf(target, zeros(0, 4))) == (0, 4)
        end

        # An aliased or mutated callback result must be copied before the input
        # buffer is reused, without changing the caller's matrix.
        aliasing = MapTargetDensity(logπ, (x::Vector{Float64}) -> x; threaded)
        @test grad_logpdf(aliasing, points) == points
        mutating = MapTargetDensity(logπ, (x::Vector{Float64}) -> (x[1] = 0.0; x); threaded)
        expected = Matrix(points)
        expected[:, 1] .= 0.0
        @test grad_logpdf(mutating, points) == expected
        @test storage == original
    end

    for distribution in (Normal(0.4, 1.3), Uniform(-1, 1))
        reference = MapReferenceDensity(distribution)
        expected = [TransportMaps.logpdf(reference, Vector(row)) for row in eachrow(points)]
        @test TransportMaps.logpdf(reference, points) ≈ expected
        expected_gradient = reduce(vcat, [permutedims(grad_logpdf(reference, Vector(row))) for row in eachrow(points)])
        @test grad_logpdf(reference, points) ≈ expected_gradient
        @test TransportMaps.logpdf(reference, points[1, 1]) ≈ TransportMaps.logpdf(distribution, points[1, 1])
        @test TransportMaps.logpdf(reference, view(points, 1, :)) ≈ expected[1]
        @test isempty(TransportMaps.logpdf(reference, zeros(0, 4)))
        @test size(grad_logpdf(reference, zeros(0, 4))) == (0, 4)
        @test TransportMaps.logpdf(reference, zeros(3, 0)) == zeros(3)
        @test size(grad_logpdf(reference, zeros(3, 0))) == (3, 0)
    end
    reference32 = MapReferenceDensity(Normal(0.0f0, 1.0f0))
    @test eltype(grad_logpdf(reference32, Float32.(points))) == Float64

    vectorized = MapTargetDensity(
        X -> vec(-sum(abs2, X; dims = 2) / 2), X -> -X;
        isvectorized = true,
    )
    @test TransportMaps.logpdf(vectorized, points) ≈ vec(-sum(abs2, points; dims = 2) / 2)
    @test grad_logpdf(vectorized, points) == -points
    @test TransportMaps.pdf(vectorized, points) ≈ exp.(TransportMaps.logpdf(vectorized, points))

    # threaded=false also applies to log-density calls; yielding would expose
    # concurrently executing callbacks if the flag were ignored.
    active = Ref(false)
    serial_logπ = x -> begin
        active[] && error("Concurrent log-density evaluation")
        active[] = true
        yield()
        result = -sum(abs2, x) / 2
        active[] = false
        result
    end
    serial = MapTargetDensity(serial_logπ, x -> -x; threaded = false)
    @test TransportMaps.logpdf(serial, points) ≈ vec(-sum(abs2, points; dims = 2) / 2)
end

@testitem "Quadrature streaming and reusable rules" begin
    using FastGaussQuadrature: gausslegendre
    using ForwardDiff

    rule = gausslegendre(100)
    original = deepcopy(rule)
    for (a, b) in ((0.0, 1.0), (1.0, -0.3), (0.5, 0.5))
        @test gaussquadrature(exp, rule, a, b) ≈ exp(b) - exp(a) atol = 1.0e-13
        @test gaussquadrature(exp, 100, a, b) ≈ gaussquadrature(exp, rule, a, b)
    end
    @test gaussquadrature(x -> cis(x), rule, 0.0, Float64(π)) ≈ 2im atol = 1.0e-13
    @test ForwardDiff.derivative(b -> gaussquadrature(exp, rule, 0.0, b), 0.7) ≈ exp(0.7) rtol = 1.0e-12
    @test gaussquadrature(exp, 0, 0.0, 1.0) == 0.0
    @test_throws DomainError gaussquadrature(exp, -1, 0.0, 1.0)
    @test_throws DimensionMismatch gaussquadrature(exp, ([0.0], [1.0, 1.0]), 0.0, 1.0)

    values = zeros(64)
    Threads.@threads for i in eachindex(values)
        values[i] = gaussquadrature(exp, rule, 0.0, i / 64)
    end
    @test values ≈ exp.((1:64) ./ 64) .- 1
    @test rule == original
end

@testitem "Root finding avoids redundant callback evaluations" begin
    for root in (-10.0, 10.0)
        calls = Float64[]
        fun(x) = (push!(calls, x); x - root)
        lower, upper = TransportMaps._inverse_bound(fun)
        @test lower <= root <= upper
        # The endpoint that remains fixed is evaluated only once.
        fixed_endpoint = root > 0 ? -1.0 : 1.0
        @test count(==(fixed_endpoint), calls) == 1
    end

    residual_calls, derivative_calls = Ref(0), Ref(0)
    fun(x) = (residual_calls[] += 1; x^3)
    derivative(x) = (derivative_calls[] += 1; 3x^2)
    @test hybridrootfinder(fun, derivative, -1.0, 1.0) == (0.0, 0.0, 0.0)
    @test residual_calls[] == derivative_calls[] == 1

    residual_calls[] = 0
    derivative_calls[] = 0
    near_root(x) = (residual_calls[] += 1; x - 1.0e-9)
    slope(x) = (derivative_calls[] += 1; 1.0)
    x, fx, _ = hybridrootfinder(near_root, slope, -1.0, 1.0)
    @test abs(fx) < 1.0e-6
    @test residual_calls[] == derivative_calls[] == 1
end
