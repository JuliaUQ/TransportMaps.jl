@testitem "Optimizer workspaces match independent objective and gradient APIs" begin
    using Distributions: Normal
    using Random
    using LinearAlgebra
    using FastGaussQuadrature: gausslegendre

    TM = TransportMaps
    rng = MersenneTwister(184)
    storage = 0.2 .* randn(rng, 14, 4)
    original = copy(storage)
    points = view(storage, 1:2:14, 1:3)
    reference = MapReferenceDensity(Normal(0.4, 1.3))
    target = MapTargetDensity(x -> -sum(abs2, x) / 2, x -> -x; threaded = false)

    for basis in (HermiteBasis(), LinearizedHermiteBasis(), GaussianWeightedHermiteBasis(), CubicSplineHermiteBasis())
        for rectifier in (Softplus(), ShiftedELU(), ExpRectifier(), IdentityRectifier())
            map = PolynomialMap(3, 3, Normal(), rectifier, basis)
            coefficients = 0.01 .* randn(rng, numbercoefficients(map))
            setcoefficients!(map, coefficients)
            # Keep IdentityRectifier diagonals away from zero on these points.
            for component in map.components
                linear = zeros(Int, component.index)
                linear[end] = 1
                j = findfirst(==(linear), TM.getmultivariateindices(component))
                component.coefficients[j] += 1.0
            end
            precomp = PrecomputedMapBasis(map, Matrix(points), randn(rng, size(points, 1)); n_quad = 5)

            # Check the newly factored tables against direct tensor-product
            # evaluations, independently of the new construction algorithm.
            nodes, _ = gausslegendre(5)
            cp = precomp.component_data[3]
            component = map[3]
            for j in eachindex(component.basisfunctions)
                z = Vector(points[2, :])
                z[3] = 0.0
                @test cp.Ψ₀[2, j] ≈ evaluate(component.basisfunctions[j], z) atol = 1.0e-14
                z[3] = nodes[3] * cp.quad_scales[2] + cp.quad_scales[2]
                @test cp.Ψ_quad[2, 3, j] ≈ evaluate(component.basisfunctions[j], z) atol = 1.0e-14
                @test cp.∂Ψ_quad[2, 3, j] ≈ partial_derivative_z(component.basisfunctions[j], z, 3) atol = 1.0e-14
            end

            work = TM._BasisWorkspace(cp)
            gradient = fill(NaN, numbercoefficients(component))
            expected_gradient = TM.objective_gradient!(component, reference, cp)
            expected_value = TM.objective(component, reference, cp)
            value = TM._sample_value_gradient!(true, gradient, cp, component.coefficients, rectifier, reference, work)
            @test value ≈ expected_value rtol = 1.0e-12
            @test gradient ≈ expected_gradient rtol = 1.0e-11 atol = 1.0e-12
            @test TM._sample_value_gradient!(true, nothing, cp, component.coefficients, rectifier, reference, work) ≈ expected_value
            fill!(gradient, NaN)
            @test TM._sample_value_gradient!(nothing, gradient, cp, component.coefficients, rectifier, reference, work) === nothing
            @test gradient ≈ expected_gradient rtol = 1.0e-11 atol = 1.0e-12

            # Reusing the workspace at different coefficients must overwrite
            # every previously computed derivative and quadrature factor.
            component.coefficients[1] += 0.02
            value = TM._sample_value_gradient!(true, gradient, cp, component.coefficients, rectifier, reference, work)
            @test value ≈ TM.objective(component, reference, cp)
            @test gradient ≈ TM.objective_gradient!(component, reference, cp) rtol = 1.0e-11 atol = 1.0e-12

            density_work = TM._density_workspace(map, precomp)
            density_gradient = fill(NaN, numbercoefficients(map))
            for δ in (0.0, 0.02)
                expected_value = TM.kldivergence(map, target, precomp; δ)
                expected_gradient = kldivergence_gradient(map, target, precomp; δ)
                value = TM._density_value_gradient!(true, density_gradient, map, target, precomp, density_work; δ)
                @test value ≈ expected_value rtol = 1.0e-12 atol = 1.0e-12
                @test density_gradient ≈ expected_gradient rtol = 1.0e-11 atol = 1.0e-12
                @test TM._density_value_gradient!(true, nothing, map, target, precomp, density_work; δ) ≈ expected_value
                fill!(density_gradient, NaN)
                @test TM._density_value_gradient!(nothing, density_gradient, map, target, precomp, density_work; δ) === nothing
                @test density_gradient ≈ expected_gradient rtol = 1.0e-11 atol = 1.0e-12
            end
            @test storage == original
        end
    end

    # Degenerate array sizes exercise BLAS' zero-size paths and buffer clearing.
    component = PolynomialMapComponent(1, 2)
    for (samples, n_quad) in ((zeros(0, 1), 5), (zeros(3, 1), 0))
        cp = PrecomputedBasis(component, samples; n_quad)
        work = TM._BasisWorkspace(cp)
        gradient = fill(NaN, numbercoefficients(component))
        value = TM._sample_value_gradient!(true, gradient, cp, component.coefficients, component.rectifier, reference, work)
        @test value ≈ TM.objective(component, reference, cp)
        @test gradient ≈ TM.objective_gradient!(component, reference, cp)
    end
end

@testitem "Combined optimizer callback requests only the required target operations" begin
    TM = TransportMaps
    map = PolynomialMap(2, 2)
    points = [0.1 0.2; -0.2 0.1; 0.3 -0.1]
    precomp = PrecomputedMapBasis(map, points, fill(1 / 3, 3); n_quad = 5)
    calls = [0, 0]
    target = MapTargetDensity(
        x -> (calls[1] += 1; -sum(abs2, x) / 2),
        x -> (calls[2] += 1; -x); threaded = false,
    )
    work = TM._density_workspace(map, precomp)
    gradient = zeros(numbercoefficients(map))
    TM._density_value_gradient!(true, nothing, map, target, precomp, work)
    @test calls == [3, 0]
    calls .= 0
    TM._density_value_gradient!(nothing, gradient, map, target, precomp, work)
    @test calls == [0, 3]
    calls .= 0
    TM._density_value_gradient!(true, gradient, map, target, precomp, work)
    @test calls == [3, 3]
end

@testitem "Combined callbacks retain optimizer compatibility" begin
    using Optim
    using Logging
    samples = reshape(collect(range(-1.0, 1.0; length = 30)), :, 1)
    target = MapTargetDensity(x -> -sum(abs2, x) / 2, x -> -x; threaded = false)
    quadrature = GaussHermiteWeights(2, 1)
    options = Optim.Options(iterations = 2)
    with_logger(NullLogger()) do
        for optimizer in (NelderMead(), BFGS(), Newton())
            result = optimize!(PolynomialMap(1, 1), samples; optimizer, options)
            @test isfinite(only(result.train_objectives))
            @test all(isfinite, Optim.minimizer(only(result.optimization_results)))
            result_density = optimize!(PolynomialMap(1, 1), target, quadrature; optimizer, options)
            @test isfinite(Optim.minimum(result_density))
            @test all(isfinite, Optim.minimizer(result_density))
        end
    end
end
