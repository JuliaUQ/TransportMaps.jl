@testsnippet MapFromDensitySetup begin
    using TransportMaps
    using Test
    using Distributions
    using LinearAlgebra
    using Random
    using Optim

end

@testitem "Regularization selects nonlinear and interaction coefficients" setup = [MapFromDensitySetup] begin
    M = PolynomialMap(2, 2, :normal, Softplus())

    nonlinear = TransportMaps._nonlinear_penalty_mask(M)
    interactions = TransportMaps._nonlinear_penalty_mask(M; interactions_only = true)

    # Component 1: [0], [1], [2]. Component 2 follows product order:
    # [0,0], [1,0], [2,0], [0,1], [1,1], [0,2].
    @test nonlinear == Bool[0, 0, 1, 0, 0, 1, 0, 1, 1]
    @test interactions == Bool[0, 0, 0, 0, 0, 0, 0, 1, 0]
end

@testitem "Smoothed L1 and L2 penalty has the expected gradient" setup = [MapFromDensitySetup] begin
    coefficients = [0.0, -0.7, 0.0, 1.2]
    penalized = Bool[0, 1, 1, 1]
    λ1, λ2, l1_eps = 0.4, 0.3, 1.0e-3

    penalty(a) = TransportMaps._regularization_penalty(a, penalized, λ1, λ2, l1_eps)
    gradient = zeros(length(coefficients))
    TransportMaps._add_regularization_gradient!(
        gradient, coefficients, penalized, λ1, λ2, l1_eps,
    )

    step = 1.0e-6
    finite_difference = [
        (penalty(coefficients + step * (eachindex(coefficients) .== i)) - penalty(coefficients - step * (eachindex(coefficients) .== i))) / (2step)
            for i in eachindex(coefficients)
    ]
    @test gradient ≈ finite_difference atol = 1.0e-7
    @test gradient[1] == 0.0
    @test gradient[3] == 0.0
    @test penalty(zeros(4)) ≈ 3λ1 * l1_eps
end

@testitem "Regularization keywords are validated and forwarded" setup = [MapFromDensitySetup] begin
    target = MapTargetDensity(x -> logpdf(Normal(), x[1]))
    quadrature = GaussHermiteWeights(3, 1)
    map = PolynomialMap(1, 2, :normal, Softplus())

    @test_throws AssertionError optimize!(deepcopy(map), target, quadrature; λ1 = -1)
    @test_throws AssertionError optimize!(deepcopy(map), target, quadrature; λ2 = -1)
    @test_throws AssertionError optimize!(deepcopy(map), target, quadrature; l1_eps = 0)
    @test_throws AssertionError optimize!(deepcopy(map), target, quadrature; δ = -1)

    precomputed = TransportMaps.PrecomputedMapBasis(map, quadrature.points, quadrature.weights)
    options = Optim.Options(iterations = 2)
    from_quadrature = deepcopy(map)
    from_precomputed = deepcopy(map)
    result_quadrature = optimize!(
        from_quadrature, target, quadrature;
        δ = 1.0e-4, λ1 = 0.2, λ2 = 0.1, l1_eps = 1.0e-4,
        interactions_only = true, options,
    )
    result_precomputed = optimize!(
        from_precomputed, target, precomputed;
        δ = 1.0e-4, λ1 = 0.2, λ2 = 0.1, l1_eps = 1.0e-4,
        interactions_only = true, options,
    )
    @test Optim.minimum(result_quadrature) ≈ Optim.minimum(result_precomputed)
    @test getcoefficients(from_quadrature) ≈ getcoefficients(from_precomputed)

    @test_throws AssertionError optimize_adaptive_transportmap(
        target, quadrature, numbercoefficients(map); initial_map = map, λ1 = -1,
    )
    @test_throws AssertionError optimize_adaptive_transportmap(
        target, quadrature, numbercoefficients(map); initial_map = map, δ = -1,
    )
end

@testitem "Map from Density" setup = [MapFromDensitySetup] begin

    @testset "KL Divergence Computation" begin
        # Test with simple 1D linear map and normal target
        M = PolynomialMap(1, 1, :normal, Softplus())
        setcoefficients!(M.components[1], [0.0, 1.0])  # Near identity map through softplus

        # Standard normal target
        target = MapTargetDensity(x -> logpdf(Normal(), x[1]))
        quadrature = GaussHermiteWeights(3, 1)

        # KL divergence should be finite for near-identity map to standard normal
        kl = TransportMaps.kldivergence(M, target, quadrature)
        @test isfinite(kl)

        # Test with shifted coefficients
        setcoefficients!(M.components[1], [1.0, 1.0])  # Different map
        kl_shifted = TransportMaps.kldivergence(M, target, quadrature)
        @test isfinite(kl_shifted)  # Should also be finite
    end

    @testset "KL Divergence Uses the Reference Density" begin
        configurations = (
            (Normal(2, 3), HermiteBasis(), GaussHermiteWeights),
            (Uniform(-1, 1), LegendreBasis(), GaussLegendreWeights),
        )

        for (reference, basis, quadrature_constructor) in configurations
            M = PolynomialMap(1, 1, reference, IdentityRectifier(), basis)
            setcoefficients!(M, [0.0, 1.0])
            target = MapTargetDensity(x -> logpdf(reference, x[1]))
            quadrature = quadrature_constructor(5, M)
            precomputed = TransportMaps.PrecomputedMapBasis(
                M, quadrature.points, quadrature.weights
            )

            @test TransportMaps.kldivergence(M, target, quadrature; δ = 0) ≈ 0.0 atol = 1.0e-12
            @test TransportMaps.kldivergence(M, target, precomputed; δ = 0) ≈ 0.0 atol = 1.0e-12
        end

        normal_map = PolynomialMap(
            1, 1, Normal(2, 3), IdentityRectifier(), HermiteBasis()
        )
        setcoefficients!(normal_map, [0.0, 1.0])
        normal_target = MapTargetDensity(x -> logpdf(Normal(2, 3), x[1]))
        normal_quadrature = GaussHermiteWeights(5, normal_map)
        exact_kl = TransportMaps.kldivergence(
            normal_map, normal_target, normal_quadrature; δ = 0
        )
        regularized_kl = TransportMaps.kldivergence(
            normal_map, normal_target, normal_quadrature; δ = 1.0e-3
        )
        @test regularized_kl != exact_kl
    end

    @testset "Non-standard Normal Reference Optimization" begin
        rng = Xoshiro(72)
        reference = Normal(2, 3)
        target_distribution = Normal(-1, 0.5)
        target = MapTargetDensity(x -> logpdf(target_distribution, x[1]))
        M = PolynomialMap(1, 1, reference, Softplus(), HermiteBasis())
        quadrature = GaussHermiteWeights(5, M)

        result = optimize!(M, target, quadrature)
        reference_samples = reshape(rand(rng, reference, 10_000), :, 1)
        mapped_samples = evaluate(M, reference_samples)

        @test Optim.converged(result)
        @test mean(mapped_samples) ≈ mean(target_distribution) atol = 0.03
        @test std(mapped_samples) ≈ std(target_distribution) atol = 0.03
        @test Optim.minimum(result) ≈ 0.0 atol = 1.0e-7
    end

    @testset "Uniform Reference Optimization" begin
        reference = Uniform(0, 1)
        target_distribution = Normal()
        target = MapTargetDensity(x -> logpdf(target_distribution, x[1]))
        M = PolynomialMap(
            1, 5, reference, Softplus(), ShiftedLegendreBasis()
        )
        quadrature = GaussLegendreWeights(5, M)

        result = optimize!(M, target, quadrature)
        mapped_points = evaluate(M, quadrature.points)
        mapped_mean = sum(quadrature.weights .* mapped_points[:, 1])
        mapped_variance = sum(
            quadrature.weights .* (mapped_points[:, 1] .- mapped_mean) .^ 2
        )

        @test Optim.converged(result)
        @test mapped_mean ≈ 0.0 atol = 0.03
        @test sqrt(mapped_variance) ≈ 1.0 atol = 0.05
        @test Optim.minimum(result) < 0.05
    end

    @testset "KL Divergence Gradient" begin
        # Test gradient computation
        M = PolynomialMap(1, 2, :normal, Softplus())
        setcoefficients!(M.components[1], [1.0, 0.1, 0.05])

        target = MapTargetDensity(x -> logpdf(Normal(), x[1]))
        quadrature = GaussHermiteWeights(3, 1)

        grad = TransportMaps.kldivergence_gradient(M, target, quadrature)
        @test length(grad) == numbercoefficients(M)
        @test all(isfinite.(grad))

        # Test finite difference approximation of gradient
        ε = 1.0e-6
        coeffs = getcoefficients(M)
        grad_fd = similar(grad)

        for i in 1:length(coeffs)
            coeffs_plus = copy(coeffs)
            coeffs_minus = copy(coeffs)
            coeffs_plus[i] += ε
            coeffs_minus[i] -= ε

            setcoefficients!(M, coeffs_plus)
            kl_plus = TransportMaps.kldivergence(M, target, quadrature)

            setcoefficients!(M, coeffs_minus)
            kl_minus = TransportMaps.kldivergence(M, target, quadrature)

            grad_fd[i] = (kl_plus - kl_minus) / (2 * ε)
        end

        # Reset coefficients
        setcoefficients!(M, coeffs)

        # Analytical and finite difference gradients should be close
        @test grad ≈ grad_fd atol = 1.0e-4
    end

    @testset "Linear Map Optimization" begin
        # Test optimization of simple linear maps

        # 2D linear map with proper initialization
        M = PolynomialMap(2, 1, :normal, Softplus())
        # Initialize close to identity map to avoid degeneracy
        setcoefficients!(M.components[1], [0.1, 0.9])      # ≈ z₁ through softplus
        setcoefficients!(M.components[2], [0.1, 0.1, 0.9]) # ≈ z₂ through softplus

        # Target: Standard bivariate normal
        target = MapTargetDensity(x -> logpdf(MvNormal(I(2)), x))
        quadrature = GaussHermiteWeights(3, 2)

        # Optimize
        result = optimize!(M, target, quadrature)
        # Check that optimization attempted some iterations or achieved reasonable result
        @test result.iterations ≥ 0  # Should at least try

        # Check that coefficients are reasonable
        coeffs = getcoefficients(M)
        @test all(isfinite.(coeffs))

        # Final KL divergence should be finite
        final_kl = TransportMaps.kldivergence(M, target, quadrature)
        @test isfinite(final_kl)
    end

    @testset "Optimization with Zero Initialization" begin
        # Test that optimization handles zero initialization gracefully

        # 1D map - simpler case
        M = PolynomialMap(1, 1, :normal, Softplus())
        target = MapTargetDensity(x -> logpdf(Normal(), x[1]))
        quadrature = GaussHermiteWeights(3, 1)

        # This should handle the zero initialization issue internally
        result = optimize!(M, target, quadrature)

        # Should at least not crash and produce finite coefficients
        coeffs = getcoefficients(M)
        @test all(isfinite.(coeffs))
    end

    @testset "Banana Density Optimization" begin
        # Test optimization with banana-shaped target density

        M = PolynomialMap(2, 2, :normal, Softplus())

        # Banana density: π(x) ∝ exp(-x₁²/2) * exp(-(x₂ - x₁²)²/2)
        banana_density = function (x)
            return (-0.5 * x[1]^2) + (-0.5 * (x[2] - x[1]^2)^2)
        end

        target = MapTargetDensity(banana_density)
        quadrature = GaussHermiteWeights(3, 2)

        # Optimize
        result = optimize!(M, target, quadrature)
        @test result.iterations > 0  # Check that optimization ran

        # Check convergence
        @test result.minimum < 10.0  # Should achieve reasonable objective value

        # Test that map produces reasonable outputs
        test_points = randn(10, 2)
        for z in eachrow(test_points)
            mapped = evaluate(M, z)
            @test all(isfinite.(mapped))
            @test length(mapped) == 2
        end
    end

    @testset "Variance Diagnostics" begin
        # Test variance diagnostic computation

        # Simple 1D case
        M = PolynomialMap(1, 1, :normal, Softplus())
        setcoefficients!(M.components[1], [0.0, 1.0])  # Near identity through softplus

        target = MapTargetDensity(x -> logpdf(Normal(), x[1]))

        # Generate test samples
        Z = randn(10, 1)

        var_diag = variance_diagnostic(M, target, Z)
        @test var_diag ≥ 0.0  # Variance must be non-negative
        @test isfinite(var_diag)

        # Test 2D case
        M2 = PolynomialMap(2, 1, :normal, Softplus())
        setcoefficients!(M2.components[1], [0.0, 1.0])       # First component through softplus
        setcoefficients!(M2.components[2], [0.0, 0.0, 1.0])  # Second component through softplus

        target2 = MapTargetDensity(x -> pdf(MvNormal(I(2)), x))
        Z2 = randn(10, 2)

        var_diag2 = variance_diagnostic(M2, target2, Z2)
        @test var_diag2 ≥ 0.0
        @test isfinite(var_diag2)
    end

    @testset "Optimization Convergence with Different Targets" begin
        # Test optimization with various target distributions

        @testset "Mixture Target" begin
            # 2D mixture of two Gaussians
            M = PolynomialMap(2, 2, :normal, Softplus())

            mixture_target = function (x)
                # Mixture of two 2D normals
                μ1 = [-1.0, -1.0]
                μ2 = [1.0, 1.0]
                Σ = I(2) * 0.5

                w1, w2 = 0.6, 0.4
                p1 = pdf(MvNormal(μ1, Σ), x)
                p2 = pdf(MvNormal(μ2, Σ), x)

                return log.(w1 * p1 + w2 * p2)
            end

            target = MapTargetDensity(mixture_target)
            quadrature = GaussHermiteWeights(3, 2)

            result = optimize!(M, target, quadrature)
            @test result.iterations > 0  # Check that optimization ran
            @test isfinite(result.minimum)

            # Test variance diagnostic for optimized map
            Z = randn(50, 2)
            var_diag = variance_diagnostic(M, target, Z)
            @test isfinite(var_diag)
            @test var_diag ≥ 0.0
        end
    end

    @testset "Edge Cases and Error Handling" begin
        # Test with very small quadrature
        M = PolynomialMap(1, 1, :normal, Softplus())
        setcoefficients!(M.components[1], [0.0, 1.0])  # Near identity through softplus

        target = MapTargetDensity(x -> pdf(Normal(), x[1]))
        quadrature_small = GaussHermiteWeights(2, 1)  # Very few points

        kl = TransportMaps.kldivergence(M, target, quadrature_small)
        @test isfinite(kl)

        grad = TransportMaps.kldivergence_gradient(M, target, quadrature_small)
        @test all(isfinite.(grad))

        # Test variance diagnostic with single sample
        Z_single = reshape([0.0], 1, 1)
        var_diag_single = variance_diagnostic(M, target, Z_single)
        # For single sample, variance can be NaN or 0, both are acceptable
        @test isnan(var_diag_single) || var_diag_single == 0.0
    end

end
