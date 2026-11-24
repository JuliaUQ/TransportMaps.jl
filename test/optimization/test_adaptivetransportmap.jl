@testsnippet AdaptiveTransportMapSetup begin
    using TransportMaps
    using Test
    using Random
    using Distributions
    using Optim
    using Statistics
    using Logging

end

@testitem "Adaptive Transport Map references" setup = [AdaptiveTransportMapSetup] begin
    samples = reshape(collect(range(-3.0, 5.0; length = 80)), :, 1)
    reference = Normal(2, 3)
    map_reference = MapReferenceDensity(reference)

    @testset "Reference-aware fitting and histories" begin
        for basis in (LinearizedHermiteBasis(), HermiteBasis())
            M, histories = optimize_adaptive_transportmap(
                samples, [3], LinearMap(), reference, Softplus(), basis;
                test_fraction = 0.0,
            )
            mapped = evaluate(M, samples)
            @test M.reference.densitytype == reference
            @test M.polynomialmap.reference.densitytype == reference
            @test TransportMaps.getbasis(M.polynomialmap[1]) isa typeof(basis)
            @test mean(mapped) ≈ mean(reference) atol = 1.0e-5
            @test std(mapped; corrected = false) ≈ std(reference) rtol = 0.02
            @test histories[1].train_objectives[end] ≈
                TransportMaps.objective(M.polynomialmap[1], map_reference, samples) / size(samples, 1)
        end

        train, validation = samples[1:60, :], samples[61:end, :]
        component, history = optimize_adaptive_transportmapcomponent(
            train, validation, 3, reference,
        )
        @test history.test_objectives[end] ≈
            TransportMaps.objective(component, map_reference, validation) / size(validation, 1)

        # The first enrichment must use the selected reference's score.
        initial = PolynomialMapComponent([[0]], Softplus(), HermiteBasis(), train)
        TransportMaps.optimize!(initial, map_reference, PrecomputedBasis(initial, train), LBFGS(), Optim.Options())
        candidate = PolynomialMapComponent([[0], [1]], Softplus(), HermiteBasis(), train)
        setcoefficients!(candidate, [getcoefficients(initial)..., 0.0])
        expected_gradient = abs(TransportMaps.objective_gradient!(candidate, map_reference, train)[end])
        _, candidate_history = optimize_adaptive_transportmapcomponent(
            train, validation, 2, reference, Softplus(), HermiteBasis(),
        )
        @test only(candidate_history.gradients[2]) ≈ expected_gradient
    end

    @testset "K-fold refit" begin
        Random.seed!(105)
        M, histories, selected_terms, _ = optimize_adaptive_transportmap(
            samples, [3], 3, LinearMap(), reference,
        )
        @test M.reference.densitytype == reference
        @test numbercoefficients(M.polynomialmap[1]) == selected_terms[1]
        @test mean(evaluate(M, samples)) ≈ mean(reference) atol = 1.0e-5
        @test all(all(isfinite, h.train_objectives) for h in histories[1])
    end

    @testset "Uniform references" begin
        for (reference, basis_type) in ((Uniform(-1, 1), LegendreBasis), (Uniform(0, 1), ShiftedLegendreBasis))
            logger = Test.TestLogger()
            M, histories = Logging.with_logger(logger) do
                optimize_adaptive_transportmap(
                    samples, [3], LinearMap(), reference; test_fraction = 0.0,
                )
            end
            @test M.reference.densitytype == reference
            @test TransportMaps.getbasis(M.polynomialmap[1]) isa basis_type
            @test all(x -> insupport(reference, x), evaluate(M, samples))
            @test all(isfinite, histories[1].train_objectives)
            @test size.(histories[1].terms, 1) == [2, 3]
            @test all(res.method isa IPNewton for res in histories[1].optimization_results)

            M_cv, _, selected_terms, _ = Logging.with_logger(logger) do
                optimize_adaptive_transportmap(samples, [3], 3, LinearMap(), reference)
            end
            @test M_cv.reference.densitytype == reference
            @test selected_terms[1] == numbercoefficients(M_cv.polynomialmap[1])
            @test all(x -> insupport(reference, x), evaluate(M_cv, samples))
            @test all(log -> log.message != "Initial guess is not an interior point", logger.logs)
        end
        @test_throws AssertionError optimize_adaptive_transportmap(samples, [1], LinearMap(), Uniform(0, 1))
        @test_throws ArgumentError optimize_adaptive_transportmap(samples, [2], LinearMap(), Uniform(0, 1); optimizer = LBFGS())
    end

    @test_throws AssertionError optimize_adaptive_transportmap(samples, [2], LinearMap(), Normal(), Softplus(), LegendreBasis())
    @test_throws AssertionError optimize_adaptive_transportmap(samples, [2], 3, LinearMap(), Uniform(0, 1), Softplus(), HermiteBasis())
    @test_throws ArgumentError optimize_adaptive_transportmap(samples, [2], LinearMap(), Uniform(2, 3))
end

@testitem "Adaptive Transport Map" setup = [AdaptiveTransportMapSetup] begin
    # Set random seed for reproducibility
    Random.seed!(42)

    # Create simple test samples from a standard normal distribution
    n_samples = 100
    n_dims = 2
    samples = randn(n_samples, n_dims)

    @testset "Basic ATM Construction and Optimization" begin
        maxterms = [2, 2]  # 2 terms max for each component
        opts = Optim.Options(iterations = 10)  # Few iterations for fast testing

        M, histories = @test_logs (
            :info, "Starting adaptive component optimization",
        ) match_mode = :any optimize_adaptive_transportmap(
            samples, maxterms; optimizer = LBFGS(), options = opts,
        )

        # Check that we get a PolynomialMap back
        @test M isa ComposedMap
        @test M.polynomialmap isa PolynomialMap
        @test M.reference.densitytype == Normal()
        @test TransportMaps.getbasis(M.polynomialmap[1]) isa LinearizedHermiteBasis

        # Check that we get histories for each component
        @test length(histories) == n_dims
        @test all(h isa OptimizationHistory for h in histories)

        # Check that each history has the right number of iterations
        @test all(length(h.train_objectives) == maxterms[i] for (i, h) in enumerate(histories))
    end

    @testset "History Tracking" begin
        maxterms = [2, 2]
        opts = Optim.Options(iterations = 5)

        M, histories = optimize_adaptive_transportmap(
            samples,
            maxterms;
            optimizer = LBFGS(),
            options = opts
        )

        for (k, history) in enumerate(histories)
            # Check that train objectives are positive and decreasing (generally)
            @test all(obj > 0 for obj in history.train_objectives)

            # Check that terms are stored
            @test all(length(term) > 0 for term in history.terms)

            # Check that gradients are stored for candidate selection
            @test length(history.gradients) == maxterms[k]
        end
    end

    @testset "Train/Test Split" begin
        maxterms = [2, 2]
        test_fraction = 0.2
        opts = Optim.Options(iterations = 5)

        M, histories = optimize_adaptive_transportmap(
            samples,
            maxterms;
            test_fraction = test_fraction,
            optimizer = LBFGS(),
            options = opts
        )

        for history in histories
            # When test_fraction > 0, test_objectives should not be empty
            # Note: test_objectives will have entries for each iteration
            @test length(history.test_objectives) > 0
        end
    end

    @testset "Different Basis Functions and Rectifiers" begin
        maxterms = [2]  # 1D only for faster testing
        samples_1d = samples[:, 1:1]
        opts = Optim.Options(iterations = 5)

        # Test with LinearizedHermiteBasis (default)
        M1, h1 = optimize_adaptive_transportmap(
            samples_1d,
            maxterms,
            LinearMap(),
            Softplus(),
            LinearizedHermiteBasis(),
            optimizer = LBFGS(),
            options = opts
        )
        @test M1 isa ComposedMap
        @test M1.polynomialmap isa PolynomialMap
        @test h1[1] isa OptimizationHistory

        # Test with HermiteBasis
        M2, h2 = optimize_adaptive_transportmap(
            samples_1d,
            maxterms,
            LinearMap(),
            ShiftedELU(),
            HermiteBasis(),
            optimizer = LBFGS(),
            options = opts
        )
        @test M2 isa ComposedMap
        @test M2.polynomialmap isa PolynomialMap
        @test h2[1] isa OptimizationHistory
    end

    @testset "Map Evaluation" begin
        maxterms = [2, 2]
        opts = Optim.Options(iterations = 5)

        M, histories = optimize_adaptive_transportmap(
            samples,
            maxterms;
            optimizer = LBFGS(),
            options = opts
        )

        # Evaluate the map on test samples
        test_sample = [0.0, 0.0]
        result = evaluate(M, test_sample)

        @test isa(result, Vector)
        @test length(result) == n_dims
    end

    @testset "K-Fold Cross Validation" begin
        maxterms = [2, 3]
        k_folds = 3
        opts = Optim.Options(iterations = 5)

        logger = Test.TestLogger()
        result = Logging.with_logger(logger) do
            optimize_adaptive_transportmap(
                samples, maxterms, k_folds; optimizer = LBFGS(), options = opts,
            )
        end
        M, fold_histories, selected_terms = result

        info_messages = [log.message for log in logger.logs if log.level == Logging.Info]
        @test "Starting k-fold adaptive component optimization" in info_messages
        @test "Selected cross-validation model" in info_messages

        @test M isa ComposedMap
        @test M.polynomialmap isa PolynomialMap
        @test length(fold_histories) == n_dims
        @test selected_terms isa Vector{Int}
        @test length(selected_terms) == n_dims
        @test all(length(component_histories) == k_folds for component_histories in fold_histories)
        @test all(all(history isa OptimizationHistory for history in component_histories) for component_histories in fold_histories)
        @test all(selected_terms[i] <= maxterms[i] for i in 1:n_dims)
        @test all(selected_terms .>= 1)
        @test all(all(length(history.train_objectives) == maxterms[i] for history in fold_histories[i]) for i in 1:n_dims)

        mean_validation = [Statistics.mean(fold_histories[i][fold].test_objectives[selected_terms[i]] for fold in 1:k_folds) for i in 1:n_dims]
        @test all(isfinite, mean_validation)
    end

end
