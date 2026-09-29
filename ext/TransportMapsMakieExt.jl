module TransportMapsMakieExt

using Makie
using TransportMaps
using Distributions: ContinuousUnivariateDistribution, Normal, pdf, quantile
using Statistics: cor
import TransportMaps: mappingplot, mappingplot!, termplot, termplot!
import TransportMaps: sampleplot, sampleplot!, transportplot, transportplot!
import TransportMaps: convergenceplot, convergenceplot!, objectiveplot, objectiveplot!
import TransportMaps: transportmap_theme, reference_target_plot, referenceplot, plotmatrix

function transportmap_theme(; fontsize = 18, titlesize = 22, size = (600, 400))
    return merge(
        Theme(;
            fontsize, size,
            Axis = (; titlefont = :regular, titlesize),
            Axis3 = (; titlefont = :regular, titlesize),
        ),
        theme_ggplot2(),
        theme_latexfonts(),
    )
end

function comparison_samples(M, samples, input_space)
    polynomial = M isa ComposedMap ? M.polynomialmap : M
    numberdimensions(polynomial) == 2 && size(samples, 2) == 2 ||
        throw(ArgumentError("reference_target_plot requires a two-dimensional map and sample matrix"))
    size(samples, 1) > 0 || throw(ArgumentError("samples must not be empty"))
    forward_space = polynomial.forwarddirection === :target ? :reference : :target
    space = input_space === :auto ? forward_space : input_space
    space in (:reference, :target) ||
        throw(ArgumentError("input_space must be :auto, :reference, or :target"))
    transformed = space === forward_space ? evaluate(M, samples) : inverse(M, samples)
    return space === :reference ? (samples, transformed) : (transformed, samples)
end

function comparison_grid(values)
    lo, hi = extrema(values)
    padding = max(hi - lo, 1.0) * 0.05
    return range(lo - padding, hi + padding; length = 100)
end

function reference_target_plot(
        M::Union{PolynomialMap, ComposedMap}, samples::AbstractMatrix{<:Real};
        input_space = :auto, density = :map, xgrid = nothing, ygrid = nothing,
        figure = (;), reference_axis = (;), target_axis = (;),
        scatter = (;), contours = (;),
    )
    reference_samples, target_samples = comparison_samples(M, samples, input_space)
    fig = Figure(; figure...)
    refax = Axis(
        fig[1, 1]; merge(
            (;
                title = "Reference", xlabel = Makie.latexstring("z_1"),
                ylabel = Makie.latexstring("z_2"), aspect = DataAspect(),
            ), reference_axis
        )...
    )
    targetax = Axis(
        fig[1, 2]; merge(
            (;
                title = "Target", xlabel = Makie.latexstring("x_1"),
                ylabel = Makie.latexstring("x_2"), aspect = DataAspect(),
            ), target_axis
        )...
    )
    style = merge((; markersize = 4, color = (:steelblue, 0.7)), scatter)
    sampleplot!(refax, reference_samples; style...)
    if density !== nothing
        xs = isnothing(xgrid) ? comparison_grid(target_samples[:, 1]) : xgrid
        ys = isnothing(ygrid) ? comparison_grid(target_samples[:, 2]) : ygrid
        density_at = density === :map ? x -> pullback(M, x) :
            x -> applicable(density, x) ? density(x) : TransportMaps.pdf(density, x)
        values = [density_at([x, y]) for x in xs, y in ys]
        contour!(
            targetax, xs, ys, values;
            merge((; levels = 8, linewidth = 2, colormap = :viridis), contours)...
        )
    end
    sampleplot!(targetax, target_samples; style...)
    return fig
end

function diagnostic_samples(M, samples, input_space)
    size(samples, 1) >= 2 || throw(ArgumentError("at least two observations are required"))
    polynomial = M isa ComposedMap ? M.polynomialmap : M
    size(samples, 2) == numberdimensions(polynomial) ||
        throw(ArgumentError("sample columns must match the map dimension"))
    all(isfinite, samples) || throw(ArgumentError("samples must be finite"))
    input_space === :reference && return samples
    input_space === :target || throw(ArgumentError("input_space must be :target or :reference"))
    return polynomial.forwarddirection === :reference ? evaluate(M, samples) : inverse(M, samples)
end

function referenceplot(
        M::Union{PolynomialMap, ComposedMap}, samples::AbstractMatrix{<:Real};
        input_space = :target, kwargs...,
    )
    polynomial = M isa ComposedMap ? M.polynomialmap : M
    Z = diagnostic_samples(M, samples, input_space)
    return referenceplot(Z; reference = polynomial.reference, kwargs...)
end

function diagnostic_values(Z, dims)
    size(Z, 1) >= 2 || throw(ArgumentError("at least two observations are required"))
    selected = collect(dims)
    !isempty(selected) && all(i -> i isa Integer && 1 <= i <= size(Z, 2), selected) &&
        length(unique(selected)) == length(selected) ||
        throw(ArgumentError("dims must contain distinct valid column indices"))
    all(isfinite, Z) || throw(ArgumentError("samples must be finite"))
    return Matrix{Float64}(Z[:, selected]), selected
end

function diagnostic_data(Z, reference, dims, qqpoints)
    values, selected = diagnostic_values(Z, dims)
    distribution = reference isa MapReferenceDensity ? reference.densitytype : reference
    distribution isa ContinuousUnivariateDistribution ||
        throw(ArgumentError("reference must be a continuous univariate distribution"))
    qqpoints isa Integer && qqpoints >= 2 || throw(ArgumentError("qqpoints must be at least two"))
    n = size(values, 1)
    probabilities = range(0.5 / n, 1 - 0.5 / n; length = min(n, qqpoints))
    theoretical = quantile.(Ref(distribution), probabilities)
    all(isfinite, theoretical) || throw(ArgumentError("reference quantiles must be finite"))
    empirical = [quantile(values[:, j], collect(probabilities)) for j in axes(values, 2)]
    correlations = cor(values; dims = 1)
    for j in axes(values, 2)
        if all(==(values[1, j]), values[:, j])
            correlations[j, :] .= NaN
            correlations[:, j] .= NaN
        end
    end
    return (; values, distribution, selected, theoretical, empirical, correlations)
end

function referenceplot(
        Z::AbstractMatrix{<:Real}; reference = Normal(), kind = :marginals,
        dims = axes(Z, 2), ncols = 2, bins = 25, qqpoints = 200,
        figure = (;), axis = (;),
    )
    kind in (:marginals, :qq, :correlation) ||
        throw(ArgumentError("kind must be :marginals, :qq, or :correlation"))
    ncols isa Integer && ncols >= 1 || throw(ArgumentError("ncols must be a positive integer"))
    bins isa Integer && bins >= 1 || throw(ArgumentError("bins must be a positive integer"))
    data = diagnostic_data(Z, reference, dims, qqpoints)
    d = length(data.selected)
    columns = min(ncols, d)
    rows = cld(d, columns)
    default_size = kind === :correlation ? (600, 500) : (600, 250rows + 60)
    fig = Figure(; merge((; size = default_size), figure)...)
    if kind === :correlation
        labels = [Makie.latexstring("z_{$j}") for j in data.selected]
        ticks = (collect(1:d), labels)
        ax = Axis(
            fig[1, 1]; merge(
                (;
                    xticks = ticks, yticks = ticks,
                    aspect = DataAspect(), yreversed = true,
                ), axis
            )...
        )
        heat = heatmap!(
            ax, 1:d, 1:d, data.correlations;
            colorrange = (-1, 1), colormap = :balance, nan_color = :gray70
        )
        Colorbar(fig[1, 2], heat; label = "Pearson correlation")
        if any(isnan, data.correlations)
            Label(fig[2, 1:2], "Gray: undefined (constant coordinate)"; fontsize = 14)
        end
        return fig
    end
    for (j, coordinate) in enumerate(data.selected)
        row, col = fldmod1(j, columns)
        values = data.values[:, j]
        title = Makie.latexstring("z_{$coordinate}")
        if kind === :marginals
            ax = Axis(fig[row, col]; merge((; title, xlabel = "Value", ylabel = "Density"), axis)...)
            hist!(ax, values; bins, normalization = :pdf, color = (:steelblue, 0.5), label = "Samples")
            lo = min(minimum(values), first(data.theoretical))
            hi = max(maximum(values), last(data.theoretical))
            # Include support endpoints, especially for bounded uniform references.
            lo = isfinite(minimum(data.distribution)) ? minimum(data.distribution) : lo
            hi = isfinite(maximum(data.distribution)) ? maximum(data.distribution) : hi
            grid = range(lo, hi; length = 300)
            density = pdf.(Ref(data.distribution), grid)
            density = [isfinite(v) ? v : NaN for v in density]
            lines!(
                ax, grid, density; color = :black,
                linewidth = 2, label = "Reference PDF"
            )
        else
            ax = Axis(
                fig[row, col]; merge(
                    (;
                        title, xlabel = "Reference quantile",
                        ylabel = "Sample quantile",
                    ), axis
                )...
            )
            scatter!(
                ax, data.theoretical, data.empirical[j]; markersize = 4,
                color = :steelblue, label = "Samples"
            )
            lo = min(first(data.theoretical), first(data.empirical[j]))
            hi = max(last(data.theoretical), last(data.empirical[j]))
            lines!(ax, [lo, hi], [lo, hi]; color = :black, linestyle = :dash, label = "Ideal agreement")
        end
        if j == 1
            Legend(fig[rows + 1, 1:columns], ax; orientation = :horizontal, framevisible = false)
        end
    end
    return fig
end

function plotmatrix_samples(M, samples, input_space, space)
    polynomial = M isa ComposedMap ? M.polynomialmap : M
    size(samples, 2) == numberdimensions(polynomial) ||
        throw(ArgumentError("sample columns must match the map dimension"))
    input_space in (:target, :reference) && space in (:target, :reference) ||
        throw(ArgumentError("input_space and space must be :target or :reference"))
    diagnostic_values(samples, axes(samples, 2))
    input_space === space && return samples
    forward_space = polynomial.forwarddirection === :target ? :reference : :target
    return input_space === forward_space ? evaluate(M, samples) : inverse(M, samples)
end

function plotmatrix(
        M::Union{PolynomialMap, ComposedMap}, samples::AbstractMatrix{<:Real};
        input_space = :target, space = :reference, dims = axes(samples, 2),
        reference = space === :reference ?
            (M isa ComposedMap ? M.polynomialmap.reference : M.reference) : nothing,
        dimlabels = nothing, kwargs...,
    )
    values = plotmatrix_samples(M, samples, input_space, space)
    _, selected = diagnostic_values(values, dims)
    prefix = space === :reference ? "z" : "x"
    labels = isnothing(dimlabels) ? [Makie.latexstring("$(prefix)_{$j}") for j in selected] : dimlabels
    return plotmatrix(values; dims = selected, reference, dimlabels = labels, kwargs...)
end

function plotmatrix(
        X::AbstractMatrix{<:Real}; dims = axes(X, 2), style = :compact,
        dimlabels = nothing, reference = nothing, bins = 25,
        color = (:steelblue, 0.4), markersize = 3, figure = (;), axis = (;),
    )
    style === :correlation && (style = :corr)
    style in (:compact, :full, :corr) ||
        throw(ArgumentError("style must be :compact, :full, or :corr"))
    bins isa Integer && bins >= 1 || throw(ArgumentError("bins must be a positive integer"))
    values, selected = diagnostic_values(X, dims)
    d = length(selected)
    distribution = reference isa MapReferenceDensity ? reference.densitytype : reference
    isnothing(distribution) || distribution isa ContinuousUnivariateDistribution ||
        throw(ArgumentError("reference must be a continuous univariate distribution"))
    prefix = isnothing(distribution) ? "x" : "z"
    labels = isnothing(dimlabels) ? [Makie.latexstring("$(prefix)_{$j}") for j in selected] : collect(dimlabels)
    length(labels) == d || throw(ArgumentError("dimlabels must have one label per selected coordinate"))
    fig = Figure(; merge((; size = (600, 600)), figure)...)
    panels = Matrix{Union{Nothing, Axis}}(nothing, d, d)
    correlations = style === :corr ? cor(values; dims = 1) : nothing
    for i in 1:d, j in 1:d
        if j > i && style !== :full
            if style === :corr
                constant = all(==(values[1, i]), values[:, i]) || all(==(values[1, j]), values[:, j])
                r = constant ? NaN : correlations[i, j]
                text = isfinite(r) ? "r = $(round(r; digits = 2))" : "undefined"
                Label(fig[i, j], text; fontsize = 16, tellwidth = false, tellheight = false)
            end
            continue
        end
        ax = Axis(
            fig[i, j]; merge(
                (;
                    xlabel = i == d ? labels[j] : "",
                    ylabel = j == 1 ? (i == j ? "Density" : labels[i]) : "",
                    xticklabelsvisible = i == d, xticksvisible = i == d,
                    yticklabelsvisible = j == 1, yticksvisible = j == 1,
                    xticklabelsize = 12, yticklabelsize = 12, xlabelsize = 18, ylabelsize = 18,
                ), axis
            )...
        )
        panels[i, j] = ax
        if i == j
            hist!(ax, values[:, j]; bins, normalization = :pdf, color)
            if !isnothing(distribution)
                probabilities = [0.5 / size(values, 1), 1 - 0.5 / size(values, 1)]
                lo, hi = quantile.(Ref(distribution), probabilities)
                all(isfinite, (lo, hi)) || throw(ArgumentError("reference quantiles must be finite"))
                lo = isfinite(minimum(distribution)) ? minimum(distribution) : min(lo, minimum(values[:, j]))
                hi = isfinite(maximum(distribution)) ? maximum(distribution) : max(hi, maximum(values[:, j]))
                grid = range(lo, hi; length = 200)
                density = [isfinite(v) ? v : NaN for v in pdf.(Ref(distribution), grid)]
                lines!(ax, grid, density; color = :black, linewidth = 2)
            end
        else
            scatter!(ax, values[:, j], values[:, i]; color, markersize)
        end
    end
    for j in 1:d
        column = [panels[i, j] for i in 1:d if panels[i, j] !== nothing]
        length(column) > 1 && linkxaxes!(column...)
        row = [panels[j, i] for i in 1:d if i != j && panels[j, i] !== nothing]
        length(row) > 1 && linkyaxes!(row...)
    end
    # Equal cell sizes keep empty upper triangles from changing panel dimensions.
    for i in 1:d
        rowsize!(fig.layout, i, Relative(1 / d))
        colsize!(fig.layout, i, Relative(1 / d))
    end
    if !isnothing(distribution)
        Legend(
            fig[d + 1, 1:d],
            [MarkerElement(; color, marker = :circle, markersize = 8), LineElement(; color = :black)],
            ["Samples", "Reference PDF"]; orientation = :horizontal, framevisible = false
        )
    end
    rowgap!(fig.layout, 6)
    colgap!(fig.layout, 6)
    return fig
end

function sample_positions(X::AbstractMatrix{<:Real}, dims)
    length(dims) == 2 || throw(ArgumentError("dims must contain two coordinate indices"))
    all(i -> i isa Integer && 1 <= i <= size(X, 2), dims) ||
        throw(ArgumentError("dims must index columns of the sample matrix"))
    i, j = dims
    return [Point2d(X[k, i], X[k, j]) for k in axes(X, 1)]
end

@recipe SamplePlot (samples,) begin
    dims = (1, 2)
    color = :steelblue
    markersize = 4
end

function Makie.plot!(p::SamplePlot)
    map!(sample_positions, p.attributes, [:samples, :dims], :positions)
    scatter!(p, p.positions; color = p.color, markersize = p.markersize)
    return p
end

@recipe TransportPlot (transport, samples) begin
    direction = :forward
    dims = (1, 2)
    color = :steelblue
    markersize = 4
end

function Makie.plot!(p::TransportPlot)
    map!(p.attributes, [:transport, :samples, :direction], :transformed) do M, X, direction
        direction === :forward && return evaluate(M, X)
        direction === :inverse && return inverse(M, X)
        throw(ArgumentError("direction must be :forward or :inverse"))
    end
    map!(sample_positions, p.attributes, [:transformed, :dims], :positions)
    scatter!(p, p.positions; color = p.color, markersize = p.markersize)
    return p
end

function mapping_positions(M, coordinates, direction, gridlines)
    transform = direction === :forward ? evaluate : direction === :inverse ? inverse :
        throw(ArgumentError("direction must be :forward or :inverse"))
    d = M isa ComposedMap ? numberdimensions(M.polynomialmap) : numberdimensions(M)
    if coordinates isa AbstractVector{<:Real}
        d == 1 || throw(ArgumentError("a coordinate vector requires a 1D map"))
        values = vec(transform(M, reshape(collect(coordinates), :, 1)))
        return Point2d.(coordinates, values)
    end
    coordinates isa Tuple && length(coordinates) == 2 && d == 2 ||
        throw(ArgumentError("a 2D map requires a tuple (xgrid, ygrid)"))
    gridlines isa Integer && gridlines >= 2 ||
        throw(ArgumentError("gridlines must be an integer of at least two"))
    xs, ys = coordinates
    all(v -> v isa AbstractVector{<:Real} && length(v) >= 2, (xs, ys)) ||
        throw(ArgumentError("each grid coordinate vector must contain at least two values"))
    points = Point2d[]
    # Transform each line separately so separators never enter the map or root solver.
    for x in range(first(xs), last(xs); length = gridlines)
        values = transform(M, hcat(fill(x, length(ys)), ys))
        append!(points, Point2d.(eachrow(values)))
        push!(points, Point2d(NaN, NaN))
    end
    for y in range(first(ys), last(ys); length = gridlines)
        values = transform(M, hcat(xs, fill(y, length(xs))))
        append!(points, Point2d.(eachrow(values)))
        push!(points, Point2d(NaN, NaN))
    end
    return points
end

@recipe MappingPlot (transport, coordinates) begin
    direction = :forward
    gridlines = 9
    color = :steelblue
    linewidth = 2
    linestyle = :solid
    label = nothing
end

function Makie.plot!(p::MappingPlot)
    map!(
        mapping_positions, p.attributes,
        [:transport, :coordinates, :direction, :gridlines], :positions
    )
    lines!(
        p, p.positions; color = p.color, linewidth = p.linewidth,
        linestyle = p.linestyle, label = p.label
    )
    return p
end

function term_values(component::PolynomialMapComponent, samples, measure)
    coefficients = getcoefficients(component)
    measure === :coefficient && return coefficients
    measure === :removal_rms ||
        throw(ArgumentError("measure must be :coefficient or :removal_rms"))
    samples isa AbstractMatrix{<:Real} && size(samples, 1) > 0 &&
        size(samples, 2) == component.index ||
        throw(ArgumentError("samples must be a nonempty N × k matrix for component k"))
    baseline = evaluate(component, samples)
    scores = zeros(length(coefficients))
    for i in eachindex(coefficients)
        iszero(coefficients[i]) && continue
        without = copy(coefficients)
        without[i] = 0.0
        reduced = PolynomialMapComponent(
            component.basisfunctions, without, component.rectifier, component.index
        )
        difference = baseline - evaluate(reduced, samples)
        scores[i] = sqrt(sum(abs2, difference) / length(difference))
    end
    return scores
end

@recipe TermPlot (component,) begin
    measure = :coefficient
    samples = nothing
    color = :steelblue
    width = 0.7
end

function Makie.plot!(p::TermPlot)
    map!(term_values, p.attributes, [:component, :samples, :measure], :values)
    map!(p.attributes, :values, :positions) do values
        Point2d.(eachindex(values), values)
    end
    barplot!(p, p.positions; direction = :x, color = p.color, width = p.width)
    return p
end

function Makie.preferred_axis_attributes(::Type{Axis}, p::TermPlot)
    indices = getmultiindexsets(p.component[])
    labels = [Makie.latexstring("(" * join(row, ",") * ")") for row in eachrow(indices)]
    return (;
        yticks = (collect(eachindex(labels)), labels),
        ylabel = Makie.latexstring("\\alpha"),
        xlabel = p.measure[] === :coefficient ? "Coefficient" : "RMS change on removal",
    )
end

@recipe ConvergencePlot (history,) begin
    show_test = false
    traincolor = :steelblue
    testcolor = :orange
    linewidth = 2
end

@recipe ObjectivePlot (result,) begin
    show_test = false
    traincolor = :steelblue
    testcolor = :orange
    linewidth = 2
end

function objective_positions(result, show_test)
    train = result.train_objectives
    test = result.test_objectives
    isempty(test) || length(test) == length(train) ||
        throw(ArgumentError("training and test objective lengths must match"))
    training = [Point2d(i, value) for (i, value) in enumerate(train)]
    testing = show_test ? [
            Point2d(i, isfinite(value) ? value : NaN)
            for (i, value) in enumerate(test)
        ] : Point2d[]
    return training, testing
end

function plot_objectives!(p, input)
    map!(objective_positions, p.attributes, [input, :show_test], [:training, :testing])
    scatterlines!(p, p.training; color = p.traincolor, linewidth = p.linewidth, label = "Training")
    scatterlines!(
        p, p.testing; color = p.testcolor, linewidth = p.linewidth,
        visible = p.show_test, label = "Test"
    )
    return p
end

function Makie.plot!(p::ConvergencePlot)
    p.history[] isa Union{OptimizationHistory, MapOptimizationResult} ||
        throw(ArgumentError("convergenceplot expects an adaptive optimization history"))
    return plot_objectives!(p, :history)
end

function Makie.plot!(p::ObjectivePlot)
    p.result[] isa OptimizationResult ||
        throw(ArgumentError("objectiveplot expects an OptimizationResult"))
    return plot_objectives!(p, :result)
end

Makie.preferred_axis_attributes(::Type{Axis}, ::ConvergencePlot) =
    (; xlabel = "Iteration", ylabel = "Objective")
Makie.preferred_axis_attributes(::Type{Axis}, ::ObjectivePlot) =
    (; xlabel = "Component", ylabel = "Objective")

# Expose the separately labelled series to axislegend, as Makie's Series does.
function Makie.get_plots(p::Union{ConvergencePlot, ObjectivePlot})
    return p.show_test[] && !isempty(p.testing[]) ? p.plots : p.plots[1:1]
end

end
