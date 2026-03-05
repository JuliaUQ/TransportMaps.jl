using TransportMaps
using Test
using Distributions
using Optim
import Mooncake

@testset "Map Density" begin

    @testset "MapTargetDensity" begin
        # Analytical gradient constructor
        target = MapTargetDensity(x -> logpdf(Normal(), x[1]))
        x = [0.0]
        g = grad_logpdf(target, x)
        @test length(g) == 1
        @test isfinite(g[1])

        # Test logpdf and pdf methods
        @test logpdf(target, [0.0]) ≈ logpdf(Normal(), 0.0)
        @test logpdf(target, 0.0) ≈ logpdf(Normal(), 0.0)
        @test pdf(target, [0.0]) ≈ pdf(Normal(), 0.0)
        @test pdf(target, 0.0) ≈ pdf(Normal(), 0.0)

        # Test logpdf with matrix input
        X = permutedims([-1.0 0.0 1.0])
        logpdfs = logpdf(target, X)
        @test length(logpdfs) == 3
        @test logpdfs[1] ≈ logpdf(Normal(), -1.0)
        @test logpdfs[2] ≈ logpdf(Normal(), 0.0)
        @test logpdfs[3] ≈ logpdf(Normal(), 1.0)

        # Test pdf with matrix input
        pdfs = pdf(target, X)
        @test length(pdfs) == 3
        @test pdfs[1] ≈ pdf(Normal(), -1.0)
        @test pdfs[2] ≈ pdf(Normal(), 0.0)
        @test pdfs[3] ≈ pdf(Normal(), 1.0)

        target_fd = MapTargetDensity(x -> logpdf(Normal(), x[1]), AutoFiniteDiff())
        g_fd = grad_logpdf(target_fd, x)
        @test length(g_fd) == 1
        @test isfinite(g_fd[1])

        target_analytical = MapTargetDensity(x -> logpdf(Normal(), x[1]), x -> [-x[1] * pdf(Normal(), x[1])])
        g_analytical = grad_logpdf(target_analytical, x)
        @test length(g_analytical) == 1
        @test isfinite(g_analytical[1])

        @testset "Gradient Types" begin
            # Test different gradient types
            gradient_types_ad = [AutoForwardDiff(), AutoFiniteDiff()]

            X = rand(5, 1)

            for type in gradient_types_ad
                t = MapTargetDensity(x -> logpdf(Normal(), x[1]), type, 1)
                @test t.ad_backend == type
                @test_nowarn grad_logpdf(t, X)
            end

        end

        @testset "Target Density with pre-computed gradient" begin
            target = MapTargetDensity(x -> logpdf(Normal(), x[1]), AutoFiniteDiff(), 1)
            @test isa(target.prepared_gradient, GradientPrep)
        end

    end

    @testset "MapTargetDensity with isvectorized flag" begin
        function logπ(X::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}})
            if X isa Vector
                return logpdf(Normal(), X[1]) + logpdf(Normal(), X[2])
            else
                n = size(X, 1)
                result = zeros(n)
                for i in 1:n
                    result[i] = logpdf(Normal(), X[i, 1]) + logpdf(Normal(), X[i, 2])
                end
                return result
            end
        end

        function grad_logπ(X::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}})
            if X isa Vector
                return [-X[1], -X[2]]
            else
                n = size(X, 1)
                result = zeros(n, 2)
                for i in 1:n
                    result[i, 1] = -X[i, 1]
                    result[i, 2] = -X[i, 2]
                end
                return result
            end
        end

        X_test = [
            -1.0 0.5
            0.0 -0.5
        ]

        @testset "Analytical gradient" begin
            target_vectorized = MapTargetDensity(logπ, grad_logπ; isvectorized=true)
            @test target_vectorized.isvectorized == true
            @test isnothing(target_vectorized.ad_backend)

            logpdf_vals = logpdf(target_vectorized, X_test)
            @test length(logpdf_vals) == 2
            @test logpdf_vals[1] ≈ logpdf(Normal(), -1.0) + logpdf(Normal(), 0.5)
            @test logpdf_vals[2] ≈ logpdf(Normal(), 0.0) + logpdf(Normal(), -0.5)

            pdf_vals = pdf(target_vectorized, X_test)
            @test length(pdf_vals) == 2
            @test pdf_vals[1] ≈ pdf(Normal(), -1.0) * pdf(Normal(), 0.5)
            @test pdf_vals[2] ≈ pdf(Normal(), 0.0) * pdf(Normal(), -0.5)

            grad_vals = grad_logpdf(target_vectorized, X_test)
            @test size(grad_vals) == (2, 2)
            @test grad_vals[1, 1] ≈ 1.0
            @test grad_vals[1, 2] ≈ -0.5
        end

        @testset "AutoForwardDiff gradient" begin
            target_vectorized = MapTargetDensity(logπ; isvectorized=true)
            @test target_vectorized.isvectorized == true
            @test target_vectorized.ad_backend == AutoForwardDiff()

            logpdf_vals = logpdf(target_vectorized, X_test)
            @test length(logpdf_vals) == 2
            @test logpdf_vals[1] ≈ logpdf(Normal(), -1.0) + logpdf(Normal(), 0.5)
            @test logpdf_vals[2] ≈ logpdf(Normal(), 0.0) + logpdf(Normal(), -0.5)

            grad_vals = grad_logpdf(target_vectorized, X_test)
            @test size(grad_vals) == (2, 2)
            @test grad_vals[1, 1] ≈ 1.0
            @test grad_vals[1, 2] ≈ -0.5

        end
        @testset "Map Optimization with vectorized density" begin
            tm = PolynomialMap(2, 1)
            quad = GaussHermiteWeights(3, 2)
            target = MapTargetDensity(logπ; isvectorized=true)
            res = optimize!(tm, target, quad)
            @test Optim.converged(res)
        end

        @testset "Threaded flag" begin
            target = MapTargetDensity(logπ)
            @test target.threaded == true

            target = MapTargetDensity(logπ, AutoMooncake())
            @test target.threaded == false

            target = MapTargetDensity(logπ, AutoMooncake(), 2)
            @test target.threaded == false
        end
    end

    @testset "MapReferenceDensity" begin
        # Auto-diff gradient constructor
        ref = MapReferenceDensity(Normal())
        x = [0.0]
        g = grad_logpdf(ref, x)
        @test length(g) == 1
        @test isfinite(g[1])

        @test_throws ErrorException MapReferenceDensity(Uniform())

        @testset "Reference Density with pre-computed gradient" begin
            reference = MapReferenceDensity(Normal(), AutoFiniteDiff(), 1)
            @test isa(reference.prepared_gradient, GradientPrep)
        end

    end

    @testset "Show Methods" begin
        target = MapTargetDensity(x -> pdf(Normal(), x[1]))
        ref = MapReferenceDensity(Normal())

        @test_nowarn display(target)
        @test_nowarn display(ref)
    end
end
