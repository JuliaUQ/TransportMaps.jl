@testsnippet PrecomputedBasisSetup begin
    using Test
    using TransportMaps
    using Random
    using LinearAlgebra
    using Optim
    using Distributions

end

@testitem "Precomputed basis" setup = [PrecomputedBasisSetup] begin
    Random.seed!(42)

    # Test parameters
    n_samples = 100
    dimension = 2
    degree = 2

    # Generate test samples
    samples = randn(n_samples, dimension)

    @testset "PrecomputedBasis Construction" begin
        M = PolynomialMap(dimension, degree, :normal, Softplus(), LinearizedHermiteBasis())
        component = M[1]
        train_samples = samples[:, 1:1]

        # Test construction with default quadrature points
        precomp = TransportMaps.PrecomputedBasis(component, train_samples)
        @test precomp.n_samples == n_samples
        @test precomp.n_basis == length(component.basisfunctions)
        @test precomp.n_quad == 64  # default
        @test size(precomp.Ψ₀) == (n_samples, precomp.n_basis)
        @test size(precomp.∂Ψ_z) == (n_samples, precomp.n_basis)
        @test size(precomp.Ψ_quad) == (n_samples, precomp.n_quad, precomp.n_basis)
        @test size(precomp.∂Ψ_quad) == (n_samples, precomp.n_quad, precomp.n_basis)

        # Test construction with custom quadrature points
        precomp_custom = TransportMaps.PrecomputedBasis(component, train_samples, n_quad = 32)
        @test precomp_custom.n_quad == 32
    end

    @testset "Numerical Accuracy - Objective" begin
        M = PolynomialMap(dimension, degree, :normal, Softplus(), LinearizedHermiteBasis())
        component = M[1]
        train_samples = samples[:, 1:1]

        # Set random coefficients
        coeffs = randn(length(component.coefficients))
        TransportMaps.setcoefficients!(component, coeffs)

        # Create precomputed basis with high quadrature accuracy
        precomp = TransportMaps.PrecomputedBasis(component, train_samples, n_quad = 100)

        # Compute objective using precomputed basis
        obj_precomp = TransportMaps.objective(component, M.reference, precomp)

        # Compute objective point by point for verification
        obj_direct = 0.0
        for i in 1:n_samples
            z = train_samples[i, :]
            M_val = evaluate(component, z)
            ∂M_val = partial_derivative_zk(component, z)
            obj_direct += -logpdf(Normal(), M_val) - log(abs(∂M_val))
        end

        # Should match to high precision with 100 quadrature points
        @test isapprox(obj_precomp, obj_direct, rtol = 1.0e-10)
    end

    @testset "Numerical Accuracy - Gradient" begin
        M = PolynomialMap(dimension, degree, :normal, Softplus(), LinearizedHermiteBasis())
        component = M[1]
        train_samples = samples[:, 1:1]

        # Set random coefficients
        coeffs = randn(length(component.coefficients))
        TransportMaps.setcoefficients!(component, coeffs)

        # Create precomputed basis with high quadrature accuracy
        precomp = TransportMaps.PrecomputedBasis(component, train_samples, n_quad = 100)

        # Compute gradient using precomputed basis
        grad_precomp = TransportMaps.objective_gradient!(component, M.reference, precomp)

        # Compute using original function
        grad_original = TransportMaps.objective_gradient!(component, M.reference, train_samples)
        @test isapprox(grad_precomp, grad_original, rtol = 1.0e-10)

        # Compute gradient point by point for verification
        n_coeffs = length(component.coefficients)
        grad_direct = zeros(Float64, n_coeffs)

        for i in 1:n_samples
            z = train_samples[i, :]
            M_val = evaluate(component, z)
            ∂M = partial_derivative_zk(component, z)
            ∂M_∂c = gradient_coefficients(component, z)
            ∂∂M_∂c = TransportMaps.partial_derivative_zk_gradient_coefficients(component, z)
            denom = max(abs(∂M), eps()) * sign(∂M)
            grad_direct .+= M_val .* ∂M_∂c .- (1.0 ./ denom) .* ∂∂M_∂c
        end

        # Should match to high precision
        @test isapprox(grad_precomp, grad_direct, rtol = 1.0e-10)
    end

    @testset "Optimization with Precomputed Basis" begin
        M = PolynomialMap(dimension, degree, :normal, Softplus(), LinearizedHermiteBasis())

        # Optimize map from samples (should use precomputation internally)
        result = optimize!(M, samples, optimizer = LBFGS(), options = Optim.Options(iterations = 10))

        # Check that optimization ran
        @test result.optimization_results[1].iterations >= 1
        @test result.optimization_results[2].iterations >= 1

        # Check that all components have been optimized
        for component in M.components
            # Coefficients should have changed from initial zeros
            @test !all(component.coefficients .== 0.0)
        end
    end

    @testset "PrecomputedBasis Memory Efficiency" begin
        M = PolynomialMap(dimension, degree, :normal, Softplus(), LinearizedHermiteBasis())
        component = M[1]
        train_samples = samples[:, 1:1]

        # Test with different quadrature sizes
        precomp_small = TransportMaps.PrecomputedBasis(component, train_samples, n_quad = 16)
        precomp_large = TransportMaps.PrecomputedBasis(component, train_samples, n_quad = 128)

        mem_small = sizeof(precomp_small.Ψ_quad) + sizeof(precomp_small.∂Ψ_quad)
        mem_large = sizeof(precomp_large.Ψ_quad) + sizeof(precomp_large.∂Ψ_quad)

        # Memory should scale linearly with n_quad
        ratio = mem_large / mem_small
        expected_ratio = 128 / 16
        @test isapprox(ratio, expected_ratio, rtol = 0.1)
    end

    @testset "Consistency Across Different Sample Sizes" begin
        M = PolynomialMap(dimension, degree, :normal, Softplus(), LinearizedHermiteBasis())
        component = M[1]

        # Set same coefficients
        coeffs = randn(length(component.coefficients))

        # Test with different sample sizes
        for n in [50, 100, 200]
            samples_n = randn(n, 1)
            TransportMaps.setcoefficients!(component, coeffs)
            precomp = TransportMaps.PrecomputedBasis(component, samples_n, n_quad = 64)

            # Objective should be roughly proportional to sample size
            obj = TransportMaps.objective(component, precomp)
            obj_per_sample = obj / n

            # Just check that it's finite and reasonable
            @test isfinite(obj_per_sample)
        end
    end

    @testset "Map from Density" begin
        M = PolynomialMap(dimension, degree, :normal, Softplus(), LinearizedHermiteBasis())
        banana_density = function (x)
            return exp(-0.5 * x[1]^2) * exp(-0.5 * (x[2] - x[1]^2)^2)
        end
        target = TransportMaps.MapTargetDensity(banana_density)

        quadrature = TransportMaps.GaussHermiteWeights(5, dimension)

        # set random initial coefficients
        setcoefficients!(M, randn(length(getcoefficients(M))))

        # Test KL divergence computation with precomputed basis
        precomp = TransportMaps.PrecomputedMapBasis(M, quadrature.points, quadrature.weights)
        kl_div = TransportMaps.kldivergence(M, target, precomp)
        kl_direct = TransportMaps.kldivergence(M, target, quadrature)
        @test isapprox(kl_div, kl_direct; rtol = 1.0e-5)

        # Test gradient of KL divergence
        grad_precomp = TransportMaps.kldivergence_gradient(M, target, precomp)
        grad_direct = TransportMaps.kldivergence_gradient(M, target, quadrature)
        @test isapprox(grad_precomp, grad_direct; rtol = 1.0e-5)
    end

    @testset "Show and Display Methods" begin
        M = PolynomialMap(dimension, degree, :normal, Softplus(), LinearizedHermiteBasis())
        component = M[1]
        train_samples = samples[:, 1:1]
        precomp = TransportMaps.PrecomputedBasis(component, train_samples)

        @test_nowarn sprint(show, precomp)
        @test_nowarn sprint(print, precomp)
        @test_nowarn display(precomp)

        quadrature = TransportMaps.GaussHermiteWeights(5, dimension)
        precomp = TransportMaps.PrecomputedMapBasis(M, quadrature.points, quadrature.weights)
        @test_nowarn sprint(show, precomp)
        @test_nowarn sprint(print, precomp)
        @test_nowarn display(precomp)
    end
end
