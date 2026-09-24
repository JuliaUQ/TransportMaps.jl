@testitem "Makie recipes" begin
    using CairoMakie
    using TransportMaps
    using Test

    import CairoMakie: Makie
    ext = Base.get_extension(TransportMaps, :TransportMapsMakieExt)
    @test ext !== nothing

    X = [1.0 2.0 3.0; 4.0 5.0 6.0]
    fig, ax, p = sampleplot(X; dims = (3, 1), color = :red)
    @test p.positions[] == [Point2d(3, 1), Point2d(6, 4)]
    Makie.update!(p; dims = (2, 3))
    @test p.positions[] == [Point2d(2, 3), Point2d(5, 6)]
    Makie.update!(p; arg1 = X[1:1, :])
    @test p.positions[] == [Point2d(2, 3)]
    @test isempty(sampleplot!(ax, zeros(0, 3)).positions[])
    @test_throws ArgumentError ext.sample_positions(X, (0, 2))
    @test_throws ArgumentError ext.sample_positions(X, (1, 4))
    @test_throws ArgumentError ext.sample_positions(X, (1,))

    M = LinearMap([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    ax2 = Axis(fig[1, 2])
    q = transportplot!(ax2, M, X; dims = (1, 3))
    Y = evaluate(M, X)
    @test q.positions[] == [Point2d(Y[k, 1], Y[k, 3]) for k in 1:2]
    Makie.update!(q; direction = :inverse)
    Y = inverse(M, X)
    @test q.positions[] == [Point2d(Y[k, 1], Y[k, 3]) for k in 1:2]
    Makie.update!(q; arg2 = X[1:1, :])
    @test length(q.positions[]) == 1
    @test_throws Exception transportplot(M, X; direction = :unknown)

    for T in (OptimizationHistory, MapOptimizationResult)
        h = T(3)
        h.train_objectives .= [3.0, 2.0, 1.0]
        h.test_objectives .= [3.5, NaN, 1.5]
        history_fig, history_ax, c = convergenceplot(h)
        @test c.training[] == [Point2d(1, 3), Point2d(2, 2), Point2d(3, 1)]
        @test isempty(c.testing[])
        @test last(Makie.get_labeled_plots(history_ax; merge = false, unique = false)) == ["Training"]
        @test history_ax.xlabel[] == "Iteration"
        Makie.update!(c; show_test = true)
        @test c.testing[][1] == Point2d(1, 3.5)
        @test isnan(c.testing[][2][2])
        @test last(Makie.get_labeled_plots(history_ax; merge = false, unique = false)) == ["Training", "Test"]
        @test_nowarn axislegend(history_ax)
        h2 = T(1)
        h2.train_objectives .= 0.5
        empty!(h2.test_objectives)
        Makie.update!(c; arg1 = h2)
        @test c.training[] == [Point2d(1, 0.5)]
        @test isempty(c.testing[])
        convergenceplot!(Axis(fig[2, T === OptimizationHistory ? 1 : 2]), h; show_test = true)
    end

    r = OptimizationResult(2)
    r.train_objectives .= [1.0, 2.0]
    r.test_objectives .= [1.5, 2.5]
    objective_fig, objective_ax, o = objectiveplot(r; show_test = true)
    @test objective_ax.xlabel[] == "Component"
    @test o.testing[] == [Point2d(1, 1.5), Point2d(2, 2.5)]
    @test_nowarn axislegend(objective_ax)
    objectiveplot!(Axis(fig[3, 1]), r; show_test = true)
    @test_throws Exception convergenceplot(r)
    @test_throws Exception objectiveplot(OptimizationHistory(0))
    pop!(r.test_objectives)
    @test_throws ArgumentError ext.objective_positions(r, true)

    mktempdir() do dir
        path = joinpath(dir, "recipes.png")
        save(path, fig)
        @test filesize(path) > 0
    end
end

@testitem "Reference and target comparison" begin
    using CairoMakie, TransportMaps, Test
    ext = Base.get_extension(TransportMaps, :TransportMapsMakieExt)
    X = [0.1 0.2; 0.3 0.4; 0.5 0.6]
    P = PolynomialMap(2, 1)
    M = ComposedMap(LinearMap([1.0, -1.0], [2.0, 3.0]), P)
    for map in (P, M), direction in (:target, :reference)
        P.forwarddirection = direction
        Y = evaluate(map, X)
        reference, target = ext.comparison_samples(map, X, :auto)
        @test reference ≈ (direction === :target ? X : Y)
        @test target ≈ (direction === :target ? Y : X)
        opposite = direction === :target ? :target : :reference
        refback, targetback = ext.comparison_samples(map, Y, opposite)
        @test refback ≈ reference
        @test targetback ≈ target
        grid = range(-1, 1; length = 5)
        fig = reference_target_plot(
            map, X; xgrid = grid, ygrid = grid,
            figure = (; size = (450, 450))
        )
        ax = contents(fig[1, 2])[1]
        @test ax.scene.plots[1][3][] ≈ [pullback(map, [x, y]) for x in grid, y in grid]
        @test ax.scene.plots[2].positions[] == Point2d.(eachrow(target))
    end
    fig = reference_target_plot(
        P, X; density = x -> x[1] + 2x[2],
        xgrid = 0:2, ygrid = 0:3, target_axis = (; title = "Known density")
    )
    ax = contents(fig[1, 2])[1]
    @test ax.title[] == "Known density"
    @test ax.scene.plots[1][3][] ≈ [x + 2y for x in 0:2, y in 0:3]
    fig = reference_target_plot(P, X; density = nothing)
    @test length(contents(fig[1, 2])[1].scene.plots) == 1
    @test_throws ArgumentError reference_target_plot(P, X; input_space = :invalid)
    @test_throws ArgumentError reference_target_plot(P, zeros(0, 2))
    @test_throws ArgumentError reference_target_plot(PolynomialMap(3, 1), zeros(2, 3))
end

@testitem "Mapping curves and grids" begin
    using CairoMakie, TransportMaps, Test
    import CairoMakie: Makie
    ext = Base.get_extension(TransportMaps, :TransportMapsMakieExt)
    M = LinearMap([1.0], [2.0])
    x = collect(-2.0:2.0)
    fig, ax, p = mappingplot(M, x; label = "Map")
    @test p.positions[] == Point2d.(x, (x .- 1) ./ 2)
    @test_nowarn axislegend(ax)
    Makie.update!(p; direction = :inverse)
    @test p.positions[] == Point2d.(x, 2 .* x .+ 1)
    Makie.update!(p; arg2 = [0.0, 1.0])
    @test p.positions[] == [Point2d(0, 1), Point2d(1, 3)]
    M2 = LinearMap([1.0, 2.0], [2.0, 3.0])
    xs, ys = collect(-1.0:1.0), collect(-2.0:2.0)
    _, _, grid = mappingplot(M2, (xs, ys); gridlines = 3)
    @test length(grid.positions[]) == 3 * (length(xs) + length(ys) + 2)
    @test grid.positions[][1:5] == Point2d.(-ones(5), (ys .- 2) ./ 3)
    @test all(isnan, grid.positions[][6])
    Makie.update!(grid; direction = :inverse)
    @test grid.positions[][1:5] == Point2d.(-ones(5), 3 .* ys .+ 2)
    @test_throws ArgumentError ext.mapping_positions(M, x, :bad, 9)
    @test_throws ArgumentError ext.mapping_positions(M2, x, :forward, 9)
    @test_throws ArgumentError ext.mapping_positions(M, (xs, ys), :forward, 9)
    @test_throws ArgumentError ext.mapping_positions(M2, (xs, ys), :forward, 1)
    @test_throws ArgumentError ext.mapping_positions(M2, ([0.0], ys), :forward, 9)
end

@testitem "Term contributions" begin
    using CairoMakie, TransportMaps, Test
    import CairoMakie: Makie
    ext = Base.get_extension(TransportMaps, :TransportMapsMakieExt)
    c = PolynomialMapComponent(1, 2, IdentityRectifier(), HermiteBasis())
    setcoefficients!(c, [1.0, -2.0, 0.5])
    X = reshape([-1.0, 0.0, 2.0], :, 1)
    before = getcoefficients(c)
    fig, ax, p = termplot(c)
    @test p.values[] == before
    @test length(ax.yticks[][2]) == 3
    Makie.update!(p; measure = :removal_rms, samples = X)
    expected = [
        sqrt(sum(abs2, before[i] .* [evaluate(b, collect(x)) for x in eachrow(X)]) / 3)
            for (i, b) in enumerate(c.basisfunctions)
    ]
    @test p.values[] ≈ expected
    @test getcoefficients(c) == before
    nonlinear = PolynomialMapComponent(1, 1, Softplus(), HermiteBasis())
    setcoefficients!(nonlinear, [0.0, 2.0])
    scores = ext.term_values(nonlinear, X, :removal_rms)
    @test scores[1] == 0
    @test scores[2] ≈ abs(Softplus()(2.0) - Softplus()(0.0)) * sqrt(sum(abs2, X) / 3)
    @test getcoefficients(nonlinear) == [0.0, 2.0]
    @test_throws ArgumentError ext.term_values(c, nothing, :removal_rms)
    @test_throws ArgumentError ext.term_values(c, zeros(0, 1), :removal_rms)
    @test_throws ArgumentError ext.term_values(c, zeros(3, 2), :removal_rms)
    @test_throws ArgumentError ext.term_values(c, X, :invalid)
    @test_nowarn termplot!(ax, c)
    @test_nowarn termplot(fig[1, 2], c; measure = :removal_rms, samples = X)
end

@testitem "Reference-space diagnostics" begin
    using CairoMakie, TransportMaps, Test, Distributions
    ext = Base.get_extension(TransportMaps, :TransportMapsMakieExt)
    Z = [-2.0 -4.0 1.0; -1.0 -2.0 1.0; 1.0 2.0 1.0; 2.0 4.0 1.0]
    data = ext.diagnostic_data(Z, Normal(2, 3), (2, 1, 3), 200)
    @test data.selected == [2, 1, 3]
    @test data.theoretical ≈ quantile.(Normal(2, 3), [0.125, 0.375, 0.625, 0.875])
    @test data.empirical[1] ≈ quantile(Z[:, 2], [0.125, 0.375, 0.625, 0.875])
    @test data.correlations[1, 2] ≈ 1
    @test isnan(data.correlations[3, 1])
    @test isnan(data.correlations[3, 3])
    uniform = ext.diagnostic_data(Z, MapReferenceDensity(Uniform()), (1,), 3)
    @test all(0 .< uniform.theoretical .< 1)
    @test length(uniform.theoretical) == 3
    for kind in (:marginals, :qq, :correlation)
        fig = referenceplot(Z; kind, dims = (2, 1, 3))
        @test fig isa Figure
        mktempdir() do dir
            path = joinpath(dir, "diagnostic.png")
            save(path, fig)
            @test filesize(path) > 0
        end
        @test referenceplot(Z; reference = Uniform(), kind, dims = (3,)) isa Figure
    end
    @test_throws ArgumentError referenceplot(Z; dims = (1, 1))
    @test_throws ArgumentError referenceplot(Z; dims = (4,))
    @test_throws ArgumentError referenceplot(Z; dims = ())
    @test_throws ArgumentError referenceplot(Z; kind = :bad)
    @test_throws ArgumentError referenceplot(Z; qqpoints = 1)
    @test_throws ArgumentError referenceplot(Z; bins = 0)
    @test_throws ArgumentError referenceplot(Z; ncols = 0)
    @test_throws ArgumentError referenceplot(Z; reference = Poisson())
    @test_throws ArgumentError referenceplot(zeros(1, 2))
    @test_throws ArgumentError referenceplot([NaN 1.0; 0.0 1.0])
    P = PolynomialMap(3, 1)
    C = ComposedMap(LinearMap([1.0, -2.0, 0.5], [2.0, 3.0, 4.0]), P)
    for M in (P, C)
        X = inverse(M, Z)
        P.forwarddirection = :reference
        @test ext.diagnostic_samples(M, X, :target) == evaluate(M, X)
        @test ext.diagnostic_samples(M, X, :target) ≈ Z atol = 1.0e-5
        P.forwarddirection = :target
        X = evaluate(M, Z)
        @test ext.diagnostic_samples(M, X, :target) == inverse(M, X)
        @test ext.diagnostic_samples(M, X, :target) ≈ Z atol = 1.0e-5
        @test ext.diagnostic_samples(M, Z, :reference) === Z
        @test referenceplot(M, X; kind = :qq, dims = (3, 1)) isa Figure
    end
    @test_throws ArgumentError referenceplot(P, Z; input_space = :bad)
    @test_throws ArgumentError referenceplot(P, zeros(4, 2))
end

@testitem "Pairwise plot matrices" begin
    using CairoMakie, TransportMaps, Test, Distributions
    ext = Base.get_extension(TransportMaps, :TransportMapsMakieExt)
    X = [-2.0 4.0 1.0; -1.0 1.0 1.0; 1.0 1.0 1.0; 2.0 4.0 1.0]
    for (style, panel_count) in ((:compact, 6), (:full, 9), (:corr, 6), (:correlation, 6))
        fig = plotmatrix(X; style)
        @test count(b -> b isa Axis, fig.content) == panel_count
        ax = contents(fig[3, 1])[1]
        @test ax.scene.plots[1][1][] == Point2f.(X[:, 1], X[:, 3])
        if style in (:corr, :correlation)
            @test contents(fig[1, 3])[1].text[] == "undefined"
            @test contents(fig[1, 2])[1].text[] == "r = 0.0"
        end
    end
    fig = plotmatrix(X; dims = (2, 1), dimlabels = ["B", "A"], reference = Normal())
    @test contents(fig[2, 1])[1].xlabel[] == "B"
    @test contents(fig[2, 1])[1].ylabel[] == "A"
    @test length(contents(fig[1, 1])[1].scene.plots) == 2
    @test referenceplot(X; dims = (2, 1), kind = :qq) isa Figure
    for style in (:compact, :full, :corr)
        @test plotmatrix(X; dims = (3,), style, reference = Uniform()) isa Figure
    end
    for kwargs in (
            (; style = :bad), (; bins = 0), (; dims = (1, 1)),
            (; dims = (4,)), (; dimlabels = ["A"]), (; reference = Poisson()),
        )
        @test_throws ArgumentError plotmatrix(X; kwargs...)
    end
    @test_throws ArgumentError plotmatrix(zeros(1, 3))
    @test_throws ArgumentError plotmatrix([Inf 1.0; 0.0 1.0])
    P = PolynomialMap(3, 1)
    M = ComposedMap(LinearMap([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]), P)
    for direction in (:target, :reference), map in (P, M)
        P.forwarddirection = direction
        forward_space = direction === :target ? :reference : :target
        output_space = direction
        Y = evaluate(map, X)
        @test ext.plotmatrix_samples(map, X, forward_space, output_space) == Y
        @test ext.plotmatrix_samples(map, Y, output_space, forward_space) == inverse(map, Y)
        @test ext.plotmatrix_samples(map, X, :target, :target) === X
        @test ext.plotmatrix_samples(map, X, :reference, :reference) === X
    end
    @test plotmatrix(M, X; input_space = :reference, dims = (3, 1)) isa Figure
    @test plotmatrix(M, X; space = :target) isa Figure
    @test_throws ArgumentError plotmatrix(M, X; space = :bad)
    @test_throws ArgumentError plotmatrix(M, zeros(4, 2))
    mktempdir() do dir
        path = joinpath(dir, "pairwise.png")
        save(path, fig)
        @test filesize(path) > 0
    end
end
