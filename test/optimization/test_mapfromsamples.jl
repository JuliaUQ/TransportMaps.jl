@testsnippet MapFromSamplesSetup begin
    using TransportMaps
    using Test
    using Random
    using Distributions
    using LinearAlgebra
    using Optim
    using Statistics

end

@testitem "Map from Samples" setup = [MapFromSamplesSetup] begin

    Random.seed!(789)

    banana_density = function (x)
        return exp(-0.5 * x[1]^2) * exp(-0.5 * (x[2] - x[1]^2)^2)
    end

    num_samples = 500

    function generate_banana_samples(n_samples::Int)
        samples = Matrix{Float64}(undef, n_samples, 2)

        count = 0
        while count < n_samples
            x1 = randn() * 2
            x2 = randn() * 3 + x1^2

            if rand() < banana_density([x1, x2]) / 0.4
                count += 1
                samples[count, :] = [x1, x2]
            end
        end

        return samples
    end

    samples_banana = generate_banana_samples(num_samples)
    M = PolynomialMap(2, 2)
    result = optimize!(M, samples_banana)

    @test result.optimization_results[1].iterations > 0  # Check that optimization ran
    @test isfinite(result.optimization_results[1].minimum)

    @test result.optimization_results[2].iterations > 0  # Check that optimization ran
    @test isfinite(result.optimization_results[2].minimum)

    samples_new = generate_banana_samples(100)
    M2 = PolynomialMap(2, 2)
    result2 = optimize!(M2, samples_new, test_fraction = 0.3)

    @test result2.optimization_results[1].iterations > 0  # Check that optimization ran
    @test isfinite(result2.optimization_results[1].minimum)

    @test result2.optimization_results[2].iterations > 0  # Check that optimization ran
    @test isfinite(result2.optimization_results[2].minimum)

    @testset "Non-standard normal reference" begin
        samples = rand(Normal(4, 2), 300, 1)
        map = PolynomialMap(1, 1, Normal(2, 3), Softplus(), HermiteBasis())
        result = optimize!(map, samples)
        mapped_samples = evaluate(map, samples)

        @test Optim.converged(result.optimization_results[1])
        @test mean(mapped_samples) ≈ 2 atol = 1.0e-6
        @test std(mapped_samples) ≈ 3 rtol = 0.02
    end

    @testset "Uniform reference" begin
        samples = reshape(collect(range(0.02, 0.98, length = 60)), :, 1)
        map = PolynomialMap(
            1, 2, Uniform(0, 1), Softplus(), ShiftedLegendreBasis()
        )
        result = optimize!(map, samples)
        mapped_samples = evaluate(map, samples)

        @test Optim.converged(result.optimization_results[1])
        @test all((0 .<= mapped_samples) .& (mapped_samples .<= 1))
        @test first(extrema(mapped_samples)) < 0.05
        @test last(extrema(mapped_samples)) > 0.95
        @test isfinite(result.train_objectives[1])

        map_lbfgs = PolynomialMap(
            1, 2, Uniform(0, 1), Softplus(), ShiftedLegendreBasis()
        )
        @test_throws ArgumentError optimize!(map_lbfgs, samples, optimizer = LBFGS())

        @testset "Analytic Hessians and in-place constraints" begin
            hessian_samples = randn(20, 2)
            hessian_map = PolynomialMap(
                2, 3, Uniform(0, 1), Softplus(), ShiftedLegendreBasis()
            )
            component = hessian_map[2]
            TransportMaps._initialize_uniform_component!(
                component, hessian_samples, Uniform(0, 1)
            )
            precomp = PrecomputedBasis(component, hessian_samples)
            coefficients = copy(component.coefficients)
            reference = MapReferenceDensity(Uniform(0, 1))

            values = fill(NaN, precomp.n_samples)
            @test TransportMaps.evaluate_M!(
                values, precomp, coefficients, component.rectifier
            ) === values
            expected_values =
                TransportMaps.evaluate_f₀(precomp, coefficients) +
                TransportMaps.evaluate_integral(
                precomp, coefficients, component.rectifier
            )
            @test values ≈ expected_values

            analytic_objective_hessian = zeros(
                precomp.n_basis, precomp.n_basis
            )
            TransportMaps._uniform_objective_hessian!(
                analytic_objective_hessian, component, precomp, coefficients
            )
            finite_difference_objective_hessian = similar(
                analytic_objective_hessian
            )
            objective_gradient! = (gradient, c) -> begin
                setcoefficients!(component, c)
                gradient .= TransportMaps.objective_gradient!(
                    component, reference, precomp
                )
            end
            TransportMaps.FiniteDiff.finite_difference_jacobian!(
                finite_difference_objective_hessian,
                objective_gradient!,
                coefficients,
            )
            @test analytic_objective_hessian ≈
                finite_difference_objective_hessian rtol = 5.0e-4 atol = 1.0e-6

            multipliers = randn(precomp.n_samples)
            analytic_constraint_hessian = zeros(
                precomp.n_basis, precomp.n_basis
            )
            TransportMaps._uniform_constraint_hessian!(
                analytic_constraint_hessian,
                component,
                precomp,
                coefficients,
                multipliers,
            )
            finite_difference_constraint_hessian = similar(
                analytic_constraint_hessian
            )
            map_gradient = Vector{Float64}(undef, precomp.n_basis)
            constraint_gradient! = (gradient, c) -> begin
                setcoefficients!(component, c)
                fill!(gradient, 0.0)
                for i in 1:precomp.n_samples
                    TransportMaps._map_coefficient_gradient!(
                        map_gradient, component, precomp, i
                    )
                    gradient .+= multipliers[i] .* map_gradient
                end
            end
            TransportMaps.FiniteDiff.finite_difference_jacobian!(
                finite_difference_constraint_hessian,
                constraint_gradient!,
                coefficients,
            )
            @test analytic_constraint_hessian ≈
                finite_difference_constraint_hessian rtol = 5.0e-4 atol = 1.0e-6
        end
    end

end
