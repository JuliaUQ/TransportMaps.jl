@testitem "Map performance regression correctness" begin
    using TransportMaps
    using Distributions
    using LinearAlgebra
    using Random

    TM = TransportMaps
    rng = MersenneTwister(918)

    # Noncontiguous inputs exercise row/prefix views and ensure that temporary
    # quadrature coordinates never leak into the caller's data.
    storage = 0.3 .* randn(rng, 16, 4)
    points = view(storage, 1:2:16, 1:3)
    original = copy(storage)
    for basis in (HermiteBasis(), LinearizedHermiteBasis(), GaussianWeightedHermiteBasis(), CubicSplineHermiteBasis())
        for rectifier in (Softplus(), ShiftedELU(), ExpRectifier())
            map = PolynomialMap(3, 2, Normal(), rectifier, basis)
            coefficients = 0.03 .* randn(rng, numbercoefficients(map))
            setcoefficients!(map, coefficients)
            precomp = PrecomputedMapBasis(map, Matrix(points), fill(1 / size(points, 1), size(points, 1)); n_quad = 100)

            values = evaluate(map, points)
            @test values ≈ evaluate(map, precomp) rtol = 1.0e-10 atol = 1.0e-12
            @test jacobian(map, points) ≈ jacobian(map, precomp) rtol = 1.0e-10
            @test gradient_coefficients(map, points) ≈ gradient_coefficients(map, precomp) rtol = 1.0e-10 atol = 1.0e-12
            @test TM.jacobian_logdet_gradient(map, points) ≈ TM.jacobian_logdet_gradient(map, precomp) rtol = 1.0e-10 atol = 1.0e-12
            @test inverse(map, values) ≈ points atol = 5.0e-6
            @test inverse(map, view(values, :, :), 2) ≈ points[:, 1:2] atol = 5.0e-6

            component = map[3]
            cp = precomp.component_data[3]
            reference = MapReferenceDensity(Normal(0.4, 1.3))
            view_precomp = PrecomputedBasis(component, points; n_quad = 100)
            @test view_precomp.∂Ψ_quad == cp.∂Ψ_quad
            @test TM.objective(component, reference, cp) ≈ TM.objective(component, reference, points) rtol = 1.0e-10
            expected_gradient = TM.objective_gradient!(component, reference, Matrix(points))
            @test TM.objective_gradient!(component, reference, cp) ≈ expected_gradient rtol = 1.0e-10 atol = 1.0e-12
            # Caller-provided buffers must be overwritten on every call.
            output = fill(NaN, numbercoefficients(component))
            TM.objective_gradient!(output, component, reference, cp)
            @test output ≈ expected_gradient rtol = 1.0e-10 atol = 1.0e-12
            TM.objective_gradient!(output, component, reference, cp)
            @test output ≈ expected_gradient rtol = 1.0e-10 atol = 1.0e-12
            @test storage == original
            @test getcoefficients(map) == coefficients
        end
    end

    # Independent finite differences check the contracted density gradient,
    # including signed quadrature weights and nonzero stability perturbations.
    map = PolynomialMap(3, 2, Normal(), Softplus(), HermiteBasis())
    coefficients = 0.04 .* randn(rng, numbercoefficients(map))
    setcoefficients!(map, coefficients)
    weights = randn(rng, size(points, 1))
    precomp = PrecomputedMapBasis(map, Matrix(points), weights; n_quad = 32)
    target = MapTargetDensity(x -> -sum(abs2, x) / 2, x -> -x)
    for δ in (0.0, 0.02)
        analytic = kldivergence_gradient(map, target, precomp; δ)
        finite_difference = similar(coefficients)
        for j in eachindex(coefficients)
            plus, minus = copy(coefficients), copy(coefficients)
            plus[j] += 1.0e-6
            minus[j] -= 1.0e-6
            setcoefficients!(map, plus)
            upper = TM.kldivergence(map, target, precomp; δ)
            setcoefficients!(map, minus)
            lower = TM.kldivergence(map, target, precomp; δ)
            finite_difference[j] = (upper - lower) / 2.0e-6
        end
        setcoefficients!(map, coefficients)
        @test analytic ≈ finite_difference rtol = 1.0e-7 atol = 1.0e-8
    end

    # Integer and Float32 coordinates retain the Float64 coefficient precision.
    component = map[3]
    for z in ([0, 1, -1], Float32[0.2, -0.1, 0.3])
        @test gradient_coefficients(component, z) ≈ gradient_coefficients(component, Float64.(z))
        @test evaluate(component, view(z, :)) ≈ evaluate(component, Float64.(z))
    end

    # The direct constructor also permits integrating a nonfinal coordinate.
    bases = [MultivariateBasis([0, 0], HermiteBasis()), MultivariateBasis([1, 2], HermiteBasis())]
    component = PolynomialMapComponent(bases, [0.4, 0.2], Softplus(), 1)
    z = [-0.3, 0.6]
    slope_argument = 0.2 * (z[2]^2 - 1)
    @test evaluate(component, z) ≈ 0.4 + Softplus()(slope_argument) * z[1]
    @test gradient_coefficients(component, z) ≈ [1.0, z[1] * TM.derivative(Softplus(), slope_argument) * (z[2]^2 - 1)]
end
