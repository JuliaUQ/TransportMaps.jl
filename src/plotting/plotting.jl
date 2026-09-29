export sampleplot, sampleplot!, transportplot, transportplot!
export convergenceplot, convergenceplot!, objectiveplot, objectiveplot!
export transportmap_theme, reference_target_plot

"""
    transportmap_theme(; fontsize = 18, titlesize = 22, size = (600, 400))

Return the TransportMaps Makie theme: ggplot2 styling, LaTeX fonts, and larger,
regular-weight axis titles. The default figure size is 600 × 400.
Load a Makie backend to enable this function, then apply it explicitly with
`set_theme!(transportmap_theme())` or `with_theme(transportmap_theme()) do ... end`.
Constructing the theme does not change Makie's current theme.
"""
function transportmap_theme end

"""
    sampleplot(X; dims = (1, 2), color = :steelblue, markersize = 4)

Plot two coordinates of an `N × d` sample matrix (one sample per row).
Load a Makie backend, e.g. `using CairoMakie`, to enable this function.
Accepts Makie's `axis` and `figure` keywords when creating a figure; for example,
`sampleplot(X; figure=(; size=(450, 450)))`.
"""
function sampleplot end

"""
    sampleplot!(ax, X; kwargs...)

Add a [`sampleplot`](@ref) to an existing Makie axis.
"""
function sampleplot! end

"""
    transportplot(M, X; direction = :forward, dims = (1, 2), kwargs...)

Plot coordinates of transformed samples. `direction=:forward` uses `evaluate(M, X)`;
`:inverse` uses `inverse(M, X)`. Rows of `X` are samples. Direction is an operation,
not a choice of target/reference space: its meaning depends on how `M` was fitted.
Supports `color` and `markersize`. Load a Makie backend to enable this function.
"""
function transportplot end

"""
    transportplot!(ax, M, X; kwargs...)

Add a [`transportplot`](@ref) to an existing Makie axis.
"""
function transportplot! end

"""
    convergenceplot(
        history; show_test = false, traincolor = :steelblue,
        testcolor = :orange, linewidth = 2
    )

Plot objectives against adaptive iteration for an `OptimizationHistory` or
`MapOptimizationResult`. Pass a completed history returned by optimization.
Test curves are opt-in because sample-based histories can store zero placeholders
when no test set was supplied. Empty/nonfinite test values are omitted.
Use `axislegend(ax)` to show the Training/Test labels.
Load a Makie backend to enable this function.
"""
function convergenceplot end

"""
    convergenceplot!(ax, history; kwargs...)

Add a [`convergenceplot`](@ref) to an existing Makie axis.
"""
function convergenceplot! end

"""
    objectiveplot(
        result; show_test = false, traincolor = :steelblue,
        testcolor = :orange, linewidth = 2
    )

Plot objectives per component of a completed `OptimizationResult`.
Test curves are opt-in; see [`convergenceplot`](@ref).
Load a Makie backend to enable this function.
"""
function objectiveplot end

"""
    objectiveplot!(ax, result; kwargs...)

Add an [`objectiveplot`](@ref) to an existing Makie axis.
"""
function objectiveplot! end

"""
    reference_target_plot(
        M, samples; input_space = :auto, density = :map,
        xgrid = nothing, ygrid = nothing, figure = (;),
        reference_axis = (;), target_axis = (;),
        scatter = (;), contours = (;)
    )

Create and return a Makie `Figure` comparing reference and target samples for a
2D `PolynomialMap` or `ComposedMap`. Rows of `samples` are observations.
With `input_space=:auto`, supply reference samples for a map fitted from density,
and target samples for a map fitted from samples. Explicit `:reference` or
`:target` also supports transporting in the opposite direction.

The target panel overlays the fitted `pullback` density by default. Pass
`density=x -> pdf(target, x)` (or a density object supporting `pdf`) to show a
known target density; `density=nothing` disables contours. `xgrid` and `ygrid`
define the target-space contour grid and default to 100 points spanning the
samples with padding. Contours represent a 2D density, not projected marginals.

Customize the figure with `figure=(; size=(600, 400))`, the axes with
`reference_axis`/`target_axis`, and plot styles with `scatter`/`contours`.
Load a Makie backend to enable this function. This figure-building helper is
static; call it again after changing the map or samples.
"""
function reference_target_plot end

export mappingplot, mappingplot!

"""
    mappingplot(
        M, coordinates; direction = :forward, gridlines = 9,
        color = :steelblue, linewidth = 2, linestyle = :solid, label = nothing
    )

Visualize a map itself. For a 1D map, pass a vector of input coordinates to plot
input against output. For a 2D map, pass `(xgrid, ygrid)` to plot a transformed
rectangular grid. The coordinate vectors control the resolution along each line;
`gridlines` controls how many lines are drawn in each direction (at least two).

`direction=:forward` uses `evaluate`; `:inverse` uses `inverse`. Coordinates
always belong to the input space of the selected operation. The meaning of
forward depends on how the map was fitted. Only 1D and 2D maps are supported.
Load a Makie backend to enable this recipe. Accepts Makie's `axis` and `figure`
keywords; add a curve to an existing axis with [`mappingplot!`](@ref).
"""
function mappingplot end

"""
    mappingplot!(ax, M, coordinates; kwargs...)

Add a map curve or transformed grid to an existing Makie axis.
See [`mappingplot`](@ref).
"""
function mappingplot! end

export termplot, termplot!

"""
    termplot(
        component; measure = :coefficient, samples = nothing,
        color = :steelblue, width = 0.7
    )

Plot one horizontal bar per basis term of a `PolynomialMapComponent`, labelled
by its multi-index. Pass `M[k]` to inspect component `k` of a polynomial map.

- `measure=:coefficient` shows the signed coefficients (no samples needed).
- `measure=:removal_rms` shows the root-mean-square change in the component output
  when each coefficient is individually set to zero, without refitting. Supply
  a nonempty `N × k` matrix with `samples=X`. All rows are equally weighted.

Removal scores depend on the evaluation samples and are not an additive or
variance decomposition: the rectifier couples terms nonlinearly. The original
component is never modified. For composed maps, inspect the polynomial component
and supply samples after applying the linear map. Samples are always in the
component's forward input coordinates, even for maps fitted from samples.

Load a Makie backend to enable this recipe. Supports Makie's `axis` and `figure`
keywords. Automatic multi-index ticks are set when the axis is created; recreate
the plot if the basis structure changes. With `termplot!`, set the existing axis's
`yticks` and labels explicitly. Removal scores cost one component evaluation per
nonzero coefficient plus one baseline evaluation for the entire sample matrix.
"""
function termplot end

"""
    termplot!(ax, component; kwargs...)

Add a term diagnostic to an existing axis; see [`termplot`](@ref).
"""
function termplot! end

export referenceplot

"""
    referenceplot(M, target_samples; kind = :marginals, input_space = :target, kwargs...)
    referenceplot(
        Z; reference = Normal(), kind = :marginals, dims = axes(Z, 2),
        ncols = 2, bins = 25, qqpoints = 200, figure = (;), axis = (;)
    )

Diagnose reference-space samples in any dimension and return a Makie `Figure`.
`kind=:marginals` overlays marginal histograms and the reference PDF; `:qq`
compares empirical and theoretical quantiles; `:correlation` shows Pearson
correlations on a fixed [-1, 1] scale. Constant coordinates have undefined
correlations, shown in gray. All observations receive equal weight.

For a `PolynomialMap` or `ComposedMap`, supply **held-out target samples**:
automatically uses `evaluate` for sample-fitted maps and `inverse` for
density-fitted maps, and reads the map's reference distribution. Set
`input_space=:reference` to skip transformation of already-mapped samples.
The matrix-only overload accepts a continuous univariate `reference` distribution
(or `MapReferenceDensity`) shared by all independent reference coordinates.

Rows are samples. Select/reorder coordinates with `dims`; `ncols` controls the
marginal/Q–Q panel layout. The default width is 600 and height grows with the
number of rows. Customize through `figure=(; size=(600, 500))` and `axis`.
At least two finite observations are required. Q–Q probabilities stay strictly
inside (0, 1); `qqpoints` limits points per panel. No samples are silently dropped.

Marginal agreement and small Pearson correlations do not prove independence or
joint distributional agreement. Use held-out data; round-tripping samples drawn
from the fitted map tests invertibility, not the quality of the fitted density.
This is a static figure-building helper; recreate it after changing inputs.
Load a Makie backend to enable this function.
"""
function referenceplot end


export plotmatrix

"""
    plotmatrix(X; dims=axes(X, 2), style=:compact, dimlabels=nothing,
               reference=nothing, bins=25, color=(:steelblue, 0.4),
               markersize=3, figure=(;), axis=(;))
    plotmatrix(M, samples; input_space=:target, space=:reference, kwargs...)

Return a scatter-plot matrix for one `N × d` sample matrix (rows are samples).
Diagonal panels show marginal PDF-normalized histograms. `style=:compact` draws
only the lower triangle; `:full` draws all pairs; `:corr` adds Pearson correlation
values in the upper triangle (`:correlation` is an alias). Undefined correlations
for constant coordinates are labelled `undefined`. All observations are plotted.

Select/reorder coordinates with `dims`. `dimlabels` supplies one label per selected
coordinate, in that order. Pass a continuous univariate `reference` distribution
(or `MapReferenceDensity`) to overlay its PDF on each diagonal panel. This assumes
the same reference marginal in every dimension. No reference is assumed for a
plain matrix; labels default to `x` coordinates, or `z` when a reference is supplied.

For a `PolynomialMap` or `ComposedMap`, `input_space` identifies the supplied
samples and `space` selects the plotted space (`:target` or `:reference`). The
helper chooses `evaluate` or `inverse` using the map's fitted direction, and
skips transformation when the spaces match. Reference-space plots automatically
overlay the map's reference PDF; pass `reference=nothing` to suppress it.
Target-space plots do not assume a known marginal density.

The default figure size is 600 × 600; customize it with `figure=(; size=(600, 600))`
and axes through `axis`. For many dimensions, select a subset to keep panels
legible. At least two finite observations are required. This static helper accepts
one matrix, not DataFrames or multiple overlaid datasets. Pairwise plots and zero
correlations do not establish joint independence. Load a Makie backend to enable it.
"""
function plotmatrix end
