@testsnippet ComposedMapSetup begin
    using TransportMaps
    using Test
    using Random

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

    pm.forwarddirection = :reference

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

@testsnippet ComposedCustomLayer begin
    using TransportMaps
    # An independently invertible nonlinear map, exercising the custom-map protocol.
    struct CubicLayer <: AbstractTransportMap end
    TransportMaps.numberdimensions(::CubicLayer) = 2
    TransportMaps.evaluate(::CubicLayer, x::AbstractVector{<:Real}) = [x[1] + x[1]^3, x[2] + x[1]^2 / 5]
    function TransportMaps.inverse(::CubicLayer, y::AbstractVector{<:Real})
        x1 = 2 / sqrt(3) * sinh(asinh(3sqrt(3) * y[1] / 2) / 3)
        return [x1, y[2] - x1^2 / 5]
    end
    TransportMaps.evaluate(M::CubicLayer, X::AbstractMatrix{<:Real}) = reduce(vcat, [permutedims(evaluate(M, x)) for x in eachrow(X)])
    TransportMaps.inverse(M::CubicLayer, X::AbstractMatrix{<:Real}) = reduce(vcat, [permutedims(inverse(M, x)) for x in eachrow(X)])
    TransportMaps.jacobian(::CubicLayer, x::AbstractVector{<:Real}) = 1 + 3x[1]^2
end

@testitem "ComposedMap layers, chain rule, and densities" setup = [ComposedCustomLayer] begin
    using TransportMaps, Test, Distributions, LinearAlgebra
    A = LinearMap([0.5, -1.0], [2.0, 3.0])
    B = CubicLayer()
    D = LinearMap([-0.2, 0.7], [0.5, 1.5])
    C = ComposedMap(A, B, D)
    x = [0.9, -0.4]
    X = [0.9 -0.4; -0.2 0.5; 0.3 0.1]
    y = evaluate(D, evaluate(B, evaluate(A, x)))
    u = (x .- A.μ) ./ A.σ
    determinant = (1 + 3u[1]^2) / prod(A.σ) / prod(D.σ)

    @test C isa AbstractTransportMap
    @test C.layers == [A, B, D]
    @test C.forwarddirection === :reference
    @test C(x) ≈ y
    @test evaluate(ComposedMap((A, B, D)), x) ≈ y
    @test evaluate(ComposedMap([A, B, D]), x) ≈ y
    @test evaluate(ComposedMap(ComposedMap(A, B), D), x) ≈ y
    @test logpdf(ComposedMap(ComposedMap(A, B), D), x) ≈ logpdf(C, x)
    @test logabsdetjacobian(ComposedMap(A, ComposedMap(B, D)), x) ≈ log(determinant)
    @test inverse(C, y) ≈ x
    @test inverse(C, evaluate(C, X)) ≈ X
    @test evaluate(C, X) ≈ reduce(vcat, [permutedims(C(r)) for r in eachrow(X)])
    @test jacobian(C, x) ≈ determinant
    @test logabsdetjacobian(C, x) ≈ log(determinant)
    @test inverse_jacobian(C, y) ≈ inv(determinant)
    @test jacobian(C, X) ≈ [jacobian(C, r) for r in eachrow(X)]
    @test inverse_jacobian(C, X) ≈ [inverse_jacobian(C, r) for r in eachrow(X)]
    @test logabsdetjacobian(C, X) ≈ log.(jacobian(C, X))

    h = 1.0e-5
    J = hcat([(C(x + h * I(2)[:, j]) - C(x - h * I(2)[:, j])) / (2h) for j in 1:2]...)
    @test jacobian(C, x) ≈ det(J) rtol = 1.0e-8
    @test pdf(C, x) ≈ prod(pdf.(Normal(), y)) * determinant
    @test logpdf(C, x) ≈ sum(logpdf.(Normal(), y)) + log(determinant)
    @test pullback(C, X) ≈ exp.(logpdf(C, X))
    target = MapTargetDensity(x -> logpdf(C, x))
    @test pushforward(C, target, y) ≈ prod(pdf.(Normal(), y))
    @test pushforward(C, target, X) ≈ [prod(pdf.(Normal(), r)) for r in eachrow(X)]

    T = ComposedMap(A, B, D; forwarddirection = :target, reference = Normal(0.3, 1.2))
    @test evaluate(T, x) ≈ y # Metadata never changes evaluation order.
    @test pullback(T, y) ≈ prod(pdf.(Normal(0.3, 1.2), x)) / determinant
    target = MapTargetDensity(x -> logpdf(T, x))
    @test pushforward(T, target, x) ≈ prod(pdf.(Normal(0.3, 1.2), x))
    @test pushforward(T, target, X) ≈ [prod(pdf.(Normal(0.3, 1.2), r)) for r in eachrow(X)]
    mvref = MvNormal([0.1, -0.2], [1.0 0.2; 0.2 2.0])
    @test pdf(ComposedMap(A, B, D; reference = mvref), x) ≈ pdf(mvref, y) * determinant

    # Density fitting direction is respected even with a non-identity polynomial.
    P = PolynomialMap(2, 1)
    P.forwarddirection = :reference
    legacy = ComposedMap(A, P)
    @test legacy.linearmap === A
    @test legacy.polynomialmap === P
    @test hasproperty(legacy, :linearmap) && hasproperty(legacy, :polynomialmap)
    @test !hasproperty(C, :polynomialmap)
    @test pullback(legacy, x) ≈ pullback(P, evaluate(A, x)) / prod(A.σ)
    original_density = pullback(legacy, x)
    P.forwarddirection = :target
    @test legacy.forwarddirection === :reference
    @test pullback(legacy, x) ≈ original_density
    retargeted = ComposedMap(A, P)
    z = inverse(retargeted, x)
    @test retargeted.forwarddirection === :target
    @test pullback(retargeted, x) ≈ prod(pdf.(Normal(), z)) / jacobian(retargeted, z)
    @test numberdimensions(ComposedMap(LinearMap(), P)) == 2
    @test evaluate(ComposedMap(LinearMap(), P), x) ≈ evaluate(P, x)
    @test inverse(ComposedMap(LinearMap(), P), evaluate(P, x)) ≈ x

    # Log densities remain usable when forming determinants would underflow.
    large = ComposedMap(LinearMap(zeros(2), fill(1.0e200, 2)))
    @test isfinite(logpdf(large, zeros(2)))
    @test logpdf(large, zeros(2)) ≈ -log(2π) - 400log(10)
    @test logabsdetjacobian(large, zeros(2)) ≈ -400log(10)
    signed = ComposedMap(LinearMap(zeros(2), [-2.0, 3.0]))
    @test jacobian(signed, x) ≈ -1 / 6
    @test logabsdetjacobian(signed, x) ≈ -log(6)
    @test pdf(signed, x) ≈ prod(pdf.(Normal(), [-x[1] / 2, x[2] / 3])) / 6

    for f in (evaluate, inverse)
        @test size(f(C, zeros(0, 2))) == (0, 2)
        @test_throws DimensionMismatch f(C, zeros(0, 3))
        @test_throws DimensionMismatch f(C, zeros(3))
    end
    for f in (jacobian, inverse_jacobian, logabsdetjacobian, pdf, logpdf, pullback)
        @test isempty(f(C, zeros(0, 2)))
        @test_throws DimensionMismatch f(C, zeros(0, 3))
        @test_throws DimensionMismatch f(C, zeros(3))
    end
    @test_throws ArgumentError ComposedMap()
    @test_throws ArgumentError ComposedMap(())
    @test_throws ArgumentError ComposedMap(Any[])
    @test_throws ArgumentError ComposedMap(42)
    @test_throws DimensionMismatch ComposedMap(A, LinearMap(3))
    @test_throws ArgumentError ComposedMap(A; forwarddirection = :unknown)
    @test_throws ArgumentError ComposedMap(A; reference = 42)
    @test_nowarn sprint(show, C)
    @test_nowarn sprint(show, MIME"text/plain"(), C)
end

@testitem "ComposedMap polynomial layers and metadata" begin
    using TransportMaps, Test, Distributions, LinearAlgebra
    maps = [
        PolynomialMap(2, 3, Normal(), Softplus(), basis)
            for basis in (LinearizedHermiteBasis(), HermiteBasis(), GaussianWeightedHermiteBasis())
    ]
    for (i, M) in enumerate(maps)
        setcoefficients!(M, 0.05 .* sin.(i .* (1:numbercoefficients(M))))
        M.forwarddirection = :reference
    end
    C = ComposedMap(maps)
    x = [0.3, -0.2]
    @test inverse(C, C(x)) ≈ x atol = 1.0e-7
    h = 1.0e-5
    J = hcat([(C(x + h * I(2)[:, j]) - C(x - h * I(2)[:, j])) / (2h) for j in 1:2]...)
    @test jacobian(C, x) ≈ det(J) rtol = 1.0e-6
    @test exp(logabsdetjacobian(C, x)) ≈ det(J) rtol = 1.0e-6
    @test C.reference === last(maps).reference
    original_reference = C.reference
    for M in maps
        M.forwarddirection = :target
    end
    @test C.reference === original_reference
    @test C.forwarddirection === :reference
    rebuilt = ComposedMap(maps)
    @test rebuilt.reference === first(maps).reference
    @test rebuilt.forwarddirection === :target
    first(maps).forwarddirection = :reference
    @test_throws ArgumentError ComposedMap(maps)
    @test C.forwarddirection === :reference
    @test rebuilt.forwarddirection === :target
    @test ComposedMap(maps; forwarddirection = :reference).forwarddirection === :reference
    # A linear layer after a polynomial is allowed, with the global reference explicit.
    @test ComposedMap(first(maps), LinearMap(2); reference = Normal()).reference.densitytype == Normal()
end

@testitem "ComposedMap Laplace layer" begin
    using TransportMaps, Test, Distributions, LinearAlgebra
    X = [-1.0 0.2; 0.2 0.8; 1.2 -0.4; 0.4 0.1]
    L = LaplaceMap(X)
    C = ComposedMap(L)
    x = [0.1, 0.2]
    @test inverse(C, C(x)) ≈ x
    @test jacobian(C, x) ≈ inv(det(L.chol))
    @test logabsdetjacobian(C, x) ≈ -log(det(L.chol))
    @test pdf(C, x) ≈ pdf(MvNormal(L), x)
end

@testitem "ComposedMap flattens nested execution in evaluation order" setup = [ComposedCustomLayer] begin
    using TransportMaps, Distributions, ForwardDiff, LinearAlgebra

    A = LinearMap([0.5, -1.0], [2.0, 3.0])
    B = CubicLayer()
    D = LinearMap([-0.2, 0.7], [0.5, 1.5])
    flat = ComposedMap(A, B, D)
    nested = ComposedMap(ComposedMap(A), ComposedMap(ComposedMap(B), D))
    tuple_nested = ComposedMap((ComposedMap(A, B), ComposedMap(D)))
    vector_nested = ComposedMap([ComposedMap(A), ComposedMap(B, D)])
    x = [0.9, -0.4]
    X = [0.9 -0.4; -0.2 0.5; 0.3 0.1]

    for C in (nested, tuple_nested, vector_nested)
        @test C.layers == [A, B, D]
        @test all(layer -> !(layer isa ComposedMap), C.layers)
        @test typeof(C) === typeof(flat)
        for points in (x, X)
            @test evaluate(C, points) ≈ evaluate(flat, points)
            @test inverse(C, evaluate(C, points)) ≈ points
            @test jacobian(C, points) ≈ jacobian(flat, points)
            @test logabsdetjacobian(C, points) ≈ logabsdetjacobian(flat, points)
            @test inverse_jacobian(C, evaluate(C, points)) ≈ inverse_jacobian(flat, evaluate(flat, points))
            @test logpdf(C, points) ≈ logpdf(flat, points)
            @test pdf(C, points) ≈ pdf(flat, points)
        end
    end

    # Flat execution must propagate ForwardDiff dual values through custom maps.
    u = evaluate(A, x)
    expected_jacobian = Diagonal(1 ./ D.σ) *
        [1 + 3u[1]^2 0; 2u[1] / 5 1] * Diagonal(1 ./ A.σ)
    @test ForwardDiff.jacobian(z -> evaluate(nested, z), x) ≈ expected_jacobian
    expected_score = -expected_jacobian' * evaluate(nested, x) +
        [6u[1] / ((1 + 3u[1]^2) * A.σ[1]), 0.0]
    @test ForwardDiff.gradient(z -> logpdf(nested, z), x) ≈ expected_score

    # Changing the caller's container cannot change an already-built chain.
    supplied_layers = Any[ComposedMap(A), B, D]
    from_vector = ComposedMap(supplied_layers)
    supplied_layers[1] = D
    empty!(supplied_layers)
    @test from_vector.layers == [A, B, D]
    @test evaluate(from_vector, x) ≈ evaluate(flat, x)
    @test logpdf(from_vector, x) ≈ logpdf(flat, x)

    # A nested dimension-free identity must remain compatible with fixed-size
    # leaves once its wrapper has been flattened.
    identity = ComposedMap(ComposedMap(LinearMap()))
    with_identity = ComposedMap(identity, nested)
    @test numberdimensions(with_identity) == 2
    @test length(with_identity.layers) == 4
    @test evaluate(with_identity, X) ≈ evaluate(flat, X)
    @test inverse(with_identity, evaluate(flat, X)) ≈ X

    # All wrappers keep the original leaf objects, including later updates.
    setparameters!(A, [0.1, 0.2], [1.2, 0.8])
    @test nested.layers[1] === A
    @test evaluate(nested, X) ≈ evaluate(D, evaluate(B, evaluate(A, X)))
end

@testitem "ComposedMap type stays fixed as a sequential chain grows" begin
    using TransportMaps, Distributions

    let layer = LinearMap([0.001, -0.002], [1.001, 0.999])
        appended = ComposedMap(layer)
        prepended = ComposedMap(layer)
        chain_type = typeof(appended)
        x = [0.3, -0.2]
        expected = evaluate(layer, x)

        for n in 2:128
            appended = ComposedMap(appended, layer)
            prepended = ComposedMap(layer, prepended)
            @test typeof(appended) === chain_type
            @test typeof(prepended) === chain_type
            @test length(appended.layers) == n
            @test length(prepended.layers) == n
            expected = evaluate(layer, expected)
        end
        @test evaluate(appended, x) ≈ expected
        @test evaluate(prepended, x) ≈ expected
        @test inverse(appended, expected) ≈ x
        @test logabsdetjacobian(appended, x) ≈ -128 * sum(log, layer.σ)
    end
end

@testitem "ComposedMap resolves nested settings at construction" begin
    using TransportMaps, Distributions

    P = PolynomialMap(2, 1, Normal(0.2, 1.3))
    Q = PolynomialMap(2, 1, Normal(-0.4, 0.8))
    inner = ComposedMap(P, Q)
    outer = ComposedMap(inner; forwarddirection = :reference)
    x = [0.3, -0.2]

    @test outer.layers == [P, Q]
    # The inner :target map supplies its first reference even when the outer
    # direction would select the last leaf of an entirely flat composition.
    @test outer.reference === P.reference
    @test logpdf(outer, x) ≈ logpdf(P.reference, evaluate(outer, x)) + logabsdetjacobian(outer, x)
    original_reference = P.reference
    P.reference = MapReferenceDensity(Normal(0.7, 1.1))
    @test inner.reference === original_reference
    @test outer.reference === original_reference
    @test ComposedMap(P, Q).reference === P.reference

    inherited = ComposedMap(inner)
    P.forwarddirection = :reference
    Q.forwarddirection = :reference
    @test inner.forwarddirection === :target
    @test inherited.forwarddirection === :target
    @test inherited.reference === original_reference
    @test outer.reference === original_reference
    updated = ComposedMap(P, Q)
    @test updated.forwarddirection === :reference
    @test updated.reference === Q.reference
    @test ComposedMap(inner).forwarddirection === :target

    # An inner explicit direction deliberately masks mixed leaf directions.
    Q.forwarddirection = :target
    explicit_reference = MapReferenceDensity(Normal(0.1, 1.7))
    overridden = ComposedMap(P, Q; forwarddirection = :target, reference = explicit_reference)
    wrapped = ComposedMap(overridden)
    @test wrapped.layers == [P, Q]
    @test wrapped.forwarddirection === :target
    @test wrapped.reference === explicit_reference
    @test logpdf(wrapped, x) ≈ logpdf(overridden, x)
    @test_throws ArgumentError ComposedMap(P, Q)
    @test inherited.forwarddirection === :target

    # Resolve endpoint settings from whole constructor arguments before
    # flattening; the outer keywords take precedence over either inner map.
    left = ComposedMap(P; forwarddirection = :target, reference = Normal(0.4, 1.2))
    right = ComposedMap(Q; forwarddirection = :target, reference = Normal(-0.7, 0.9))
    combined = ComposedMap(left, right)
    @test combined.layers == [P, Q]
    @test combined.forwarddirection === :target
    @test combined.reference === left.reference
    reversed_direction = ComposedMap(left, right; forwarddirection = :reference)
    @test reversed_direction.forwarddirection === :reference
    @test reversed_direction.reference === right.reference
    outer_override = ComposedMap(left, right; forwarddirection = :reference, reference = explicit_reference)
    @test outer_override.forwarddirection === :reference
    @test outer_override.reference === explicit_reference
    conflicting = ComposedMap(Q; forwarddirection = :reference)
    @test_throws ArgumentError ComposedMap(left, conflicting)
    @test ComposedMap(left, conflicting; forwarddirection = :target).reference === left.reference
end

@testitem "ComposedMap composition operator" begin
    using TransportMaps, Test

    A = LinearMap([0.0], [2.0]) # A(x) = x / 2
    B = LinearMap([1.0], [1.0]) # B(x) = x - 1
    x = [3.0]

    C = A ∘ B

    @test C isa ComposedMap
    @test C(x) ≈ [1.0]           # A(B(3)) = (3 - 1) / 2
    @test (B ∘ A)(x) ≈ [0.5]     # B(A(3)) = 3 / 2 - 1
    @test C(x) ≈ evaluate(A, evaluate(B, x))
    @test inverse(C, C(x)) ≈ x
end
