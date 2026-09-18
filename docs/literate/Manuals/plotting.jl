# # Plotting with Makie
#
# Plotting is optional. Install a [Makie](https://docs.makie.org/stable/) backend in your application environment:
#
# ```julia
# using Pkg
# Pkg.add(["CairoMakie", "LaTeXStrings"])
# ```
#
# Loading the backend activates the TransportMaps Makie extension. The recipes work
# with Makie 0.24.13 or later in the 0.24 series;
# [CairoMakie](https://docs.makie.org/dev/explanations/backends/cairomakie) 0.15 is suitable for saved figures.
#
# The shared [`transportmap_theme`](@ref) combines ggplot2 styling with LaTeX fonts,
# a base font size of 18, regular-weight titles of size 22, and a default figure
# size of 600 × 400. The theme is opt-in and can be customized with `fontsize`,
# `titlesize`, and `size` keyword arguments. Use `L"..."` strings
# for mathematical labels; interpolate values in these strings with `%$value`.
#
# ## Samples and transformations
#
# Samples use the same convention as `evaluate`: an `N × d` matrix with one sample
# per row. Choose two columns with `dims` (including for higher-dimensional data).
#
using TransportMaps, CairoMakie, LaTeXStrings, Random
set_theme!(transportmap_theme())

X = randn(MersenneTwister(42), 500, 3)
M = LinearMap([1.0, -1.0, 0.0], [2.0, 0.5, 1.0])

fig = Figure(size = (600, 350))
input_ax = Axis(fig[1, 1]; title = "Input", xlabel = L"x_1", ylabel = L"x_3")
output_ax = Axis(fig[1, 2]; title = "Transformed", xlabel = L"y_1", ylabel = L"y_3")
sampleplot!(input_ax, X; dims = (1, 3), color = (:steelblue, 0.7))
transportplot!(
    output_ax, M, X; dims = (1, 3), direction = :forward,
    color = (:steelblue, 0.7)
)
linkaxes!(input_ax, output_ax)
#md save("plotting-transformation.svg", fig); nothing # hide
# ![Samples before and after a linear transformation, on matching axis scales](plotting-transformation.svg)
#
# `direction=:forward` calls `evaluate`, while `direction=:inverse` calls `inverse`.
# For maps fitted from samples, forward evaluation maps target samples into reference
# space. No samples are generated implicitly by the recipes.
#
# The functions without `!` create a figure and axis:
#
# ```julia
# fig, ax, plt = sampleplot(X; dims=(1, 2), figure=(; size=(450, 450)))
# ```
#
# ## Visualizing the mapping itself
#
# [`mappingplot`](@ref) draws a 1D map as a function curve and a 2D map as a
# transformed rectangular grid. As with `transportplot`, `direction` selects
# `evaluate` or `inverse`; it does not automatically select reference or target space.
#
# ### One dimension: CDF and inverse CDF
#
# For a continuous target with CDF ``F``, the increasing map to a uniform reference
# is ``S(x)=F(x)``, and the reverse map is ``T(u)=F^{-1}(u)``.
# Here we choose a logistic target, fit a map from its density, and compare both
# directions with the exact functions. You can replace `distribution` with another
# smooth univariate distribution supported by the optimizer.

using Distributions

distribution = Logistic(0, 1)
target_1d = MapTargetDensity(x -> logpdf(distribution, x[1]))
map_1d = PolynomialMap(1, 9, Uniform(0, 1), Softplus(), ShiftedLegendreBasis())
result_1d = optimize!(map_1d, target_1d, GaussLegendreWeights(6, map_1d))
#md nothing # hide

# Density-based fitting constructs ``T: u \mapsto x``, so `evaluate` approximates
# the quantile function and `inverse` approximates the CDF. For a map fitted from
# samples these operations are reversed. Finite-degree maps are approximations;
# we omit the uniform endpoints, where this target's exact quantile diverges.

x = range(-4, 4; length = 300)
u = range(0.02, 0.98; length = 300)
cdf_fig = Figure(size = (600, 350))
cdf_ax = Axis(cdf_fig[1, 1]; title = "To uniform", xlabel = L"x", ylabel = L"u")
quantile_ax = Axis(cdf_fig[1, 2]; title = "From uniform", xlabel = L"u", ylabel = L"x")
mappingplot!(cdf_ax, map_1d, x; direction = :inverse, label = "Fitted map")
lines!(
    cdf_ax, x, cdf.(distribution, x); color = :black, linestyle = :dash,
    label = "Exact CDF / quantile"
)
mappingplot!(quantile_ax, map_1d, u; label = "Fitted map")
lines!(quantile_ax, u, quantile.(distribution, u); color = :black, linestyle = :dash)
Legend(cdf_fig[2, 1:2], cdf_ax; orientation = :horizontal, framevisible = false)
#md save("plotting-cdf.svg", cdf_fig); nothing # hide
# ![Fitted logistic-to-uniform map compared with the CDF, and its reverse compared with the quantile function](plotting-cdf.svg)
#
# ### Two dimensions: transformed grid
#
# Pass `(xgrid, ygrid)` to draw a transformed grid. These vectors specify the
# resolution along each curve; `gridlines` sets the number of curves in each
# direction. For example, using `normal_map` from
# [Banana: Map from Density](../Examples/banana_mapfromdensity.md):
#
# ```julia
# grid = range(-2, 2; length=100)
# fig, ax, plt = mappingplot(normal_map, (grid, grid); gridlines=9,
#     figure=(; size=(600, 400)),
#     axis=(; xlabel=L"x_1", ylabel=L"x_2"))
# ```
#
# The full example compares the input grid with its image:
#
# ![A regular reference grid and its curved image under the fitted banana map](../Examples/banana-density-mapping.svg)
#
# Use `direction=:inverse` with a grid in the forward map's output space to
# visualize the inverse transformation. Grid plots currently support 2D maps;
# they do not represent projections or slices of higher-dimensional maps.
#
# ## Reference and target comparison
#
# [`reference_target_plot`](@ref) builds a two-panel figure for a 2D fitted map.
# It automatically chooses the mapping direction: supply reference samples for
# a map fitted from density, or target samples for a map fitted from samples.
# The target panel shows the fitted density by default; pass a known target
# density with `density=target` or disable contours with `density=nothing`.
#
# The following figure is reused from [Banana: Map from Density](../Examples/banana_mapfromdensity.md).
# Using the fitted `normal_map`, reference samples, and target density from that example:
#
# ```julia
# fig = reference_target_plot(normal_map, normal_ref_samples;
#     density=target,
#     xgrid=range(-4, 4; length=120),
#     ygrid=range(-3, 7; length=120),
#     reference_axis=(; limits=((-4, 4), (-4, 4))))
# ```
#
# ![Normal reference samples and transported banana samples over the known target density](../Examples/banana-density-normal-reference.svg)
#
# For the corresponding sample-based construction, see
# [Banana: Map from Samples](../Examples/banana_mapfromsamples.md).
#
# Set `input_space=:reference` or `:target` to choose the input space explicitly,
# including when using the map in the opposite direction. Customize axes through
# `reference_axis=(; title="Reference")` and `target_axis=(; title="Target")`.
#
# ## Reference-space diagnostics in higher dimensions
#
# [`referenceplot`](@ref) works with any number of coordinates. Marginal histograms
# and Q–Q plots compare each coordinate with the reference distribution; a Pearson
# correlation matrix checks for residual linear dependence between coordinates.
# The reference used by TransportMaps is a product of identical univariate distributions,
# so its population correlation matrix is the identity when second moments exist.
#
# We fit a four-dimensional map to a target with nonlinear dependencies and
# diagnose a separate held-out sample. The construction below is only for generating
# an example target; the diagnostic helper does not require its density.

function diagnostic_target_samples(rng, n)
    U = randn(rng, n, 4)
    return hcat(
        U[:, 1], U[:, 2] .+ 0.4 .* U[:, 1] .^ 2,
        U[:, 3] .+ 0.6 .* U[:, 2], U[:, 4] .+ 0.5 .* U[:, 1] .* U[:, 3]
    )
end

rng = MersenneTwister(2026)
training_samples = diagnostic_target_samples(rng, 600)
validation_samples = diagnostic_target_samples(rng, 1000)
linear_map = LinearMap(training_samples)
polynomial_map = PolynomialMap(4, 2, Normal(), Softplus(), HermiteBasis())
optimize!(polynomial_map, training_samples, linear_map)
fitted_map = ComposedMap(linear_map, polynomial_map)
#md nothing # hide

# Pass target samples directly. For a map fitted from samples, the helper calls
# `evaluate`; for a map fitted from density it calls `inverse`. It always uses the
# map's configured reference, including uniform or nonstandard normal references.

marginal_fig = referenceplot(fitted_map, validation_samples; kind = :marginals)
#md save("plotting-reference-marginals.svg", marginal_fig); nothing # hide
# ![Four mapped validation marginals overlaid with the standard-normal reference PDF](plotting-reference-marginals.svg)
#
# Histograms show shifts, incorrect spread, and out-of-support observations.
# Q–Q plots provide a bin-free comparison: agreement follows the dashed diagonal,
# while deviations reveal differences in the central region or tails.
# Already-mapped samples can be supplied directly, avoiding repeated transformation:

Z = evaluate(fitted_map, validation_samples)
qq_fig = referenceplot(Z; reference = Normal(), kind = :qq)
#md save("plotting-reference-qq.svg", qq_fig); nothing # hide
# ![Reference versus empirical quantiles for four mapped validation coordinates](plotting-reference-qq.svg)
#
# The correlation color scale is fixed at ``[-1,1]`` across figures. A constant
# coordinate has undefined correlations, which are shown in gray rather than as zero.

correlation_fig = referenceplot(Z; reference = Normal(), kind = :correlation)
#md save("plotting-reference-correlation.svg", correlation_fig); nothing # hide
# ![Pearson correlations between four mapped validation coordinates](plotting-reference-correlation.svg)
#
# For larger maps, select and reorder coordinates and control the panel layout:
#
# ```julia
# referenceplot(fitted_map, validation_samples; kind=:qq, dims=(1, 3, 4), ncols=2)
# referenceplot(Z; reference=Normal(), kind=:marginals, dims=(2, 4), bins=30)
# ```
#
# Use held-out target observations to assess fit quality. Generating samples from
# the fitted map and mapping them back only tests the round trip. Even perfect
# marginal agreement and zero Pearson correlations do not establish independence:
# nonlinear dependence can remain. These plots are diagnostics, not goodness-of-fit tests.
#
# ## Pairwise scatter-plot matrices
#
# [`plotmatrix`](@ref) provides a matrix-only view inspired by
# [PlotMatrix.jl](https://github.com/lukasfritsch/PlotMatrix.jl): marginal histograms
# on the diagonal and scatter plots for pairs of coordinates. It accepts one
# sample matrix per figure and shares coordinate limits between related panels.
#
# Reusing the four-dimensional validation samples above, the target-space view
# reveals nonlinear dependencies that a correlation matrix alone can miss:

pairwise_target_fig = plotmatrix(validation_samples; style = :corr)
#md save("plotting-pairwise-target.svg", pairwise_target_fig); nothing # hide
# ![Four-dimensional target samples with marginal histograms, pairwise scatter plots, and Pearson correlations](plotting-pairwise-target.svg)
#
# Passing a fitted map transforms target samples to reference space and overlays
# the configured reference PDF on each diagonal histogram. This works for maps
# fitted from samples or density, including composed maps:

pairwise_reference_fig = plotmatrix(fitted_map, validation_samples; style = :corr)
#md save("plotting-pairwise-reference.svg", pairwise_reference_fig); nothing # hide
# ![Mapped reference samples with pairwise scatter plots and marginal reference PDF overlays](plotting-pairwise-reference.svg)
#
# `style=:compact` (the default) leaves the upper triangle empty; `:full` plots
# every pair; `:corr` displays Pearson correlations above the diagonal. Constant
# coordinates have correlations labelled `undefined`. Histograms use all samples
# and PDF normalization. Reference PDFs are shown in black when requested.
#
# For high-dimensional maps, select a manageable subset. Labels retain original
# coordinate indices, or supply one `dimlabels` entry per selected coordinate:
#
# ```julia
# plotmatrix(validation_samples; dims=(1, 3, 4), style=:full)
# plotmatrix(Z; reference=Normal(), dims=(2, 4), dimlabels=[L"z_2", L"z_4"])
# plotmatrix(fitted_map, validation_samples; space=:target) # no transformation
# plotmatrix(fitted_map, Z; input_space=:reference)        # already mapped
# plotmatrix(fitted_map, Z; input_space=:reference, space=:target) # generate target samples
# ```
#
# Customize with `figure=(; size=(600, 600))`, `axis`, `bins`, `markersize`, and
# `color`. Pairwise views complement the marginal and Q–Q diagnostics; they still
# cannot rule out higher-order dependence involving three or more coordinates.
#
# ## Term and coefficient contributions
#
# [`termplot`](@ref) labels each term by its multi-index ``\alpha``. By default,
# it shows signed coefficients of the polynomial parameterization. Coefficient
# magnitudes alone do not measure the term's effect on the final map: basis
# scaling matters, and the rectifier couples the terms nonlinearly.
#
# To inspect the final map, use `measure=:removal_rms`. For each coefficient,
# this computes the RMS difference between the full component and the same
# component with that coefficient set to zero, without refitting:
#
# ```math
# r_\alpha = \sqrt{\frac{1}{N}\sum_{i=1}^N
# \left[M^k(z_i)-M^k_{a_\alpha=0}(z_i)\right]^2}.
# ```
#
# Using the fitted map and reference samples from
# [Banana: Map from Density](../Examples/banana_mapfromdensity.md):
#
# ```julia
# fig, ax, plt = termplot(normal_map[2])
# fig, ax, plt = termplot(normal_map[2]; measure=:removal_rms,
#     samples=normal_ref_samples[1:300, :], figure=(; size=(600, 350)))
# ```
#
# ![Coefficients and removal effects for each term of the second banana-map component](../Examples/banana-density-term-contributions.svg)
#
# Scores depend on the supplied sample distribution; they are not percentages,
# variance contributions, or an additive decomposition. A removed coefficient's
# sign is visible in the coefficient plot, while RMS changes are nonnegative.
# Both diagnostics leave the fitted component unchanged.
#
# Supply an `N × k` matrix in the component's forward input coordinates. For
# sample-fitted maps these are target coordinates; for density-fitted maps they
# are reference coordinates. When inspecting a composed map, first apply its
# linear map and then pass the first `k` columns to the polynomial component:
#
# ```julia
# Z = evaluate(composed_map.linearmap, target_samples)
# termplot(composed_map.polynomialmap[k]; measure=:removal_rms, samples=Z[:, 1:k])
# ```
#
# ## Optimization results
#
# For completed adaptive histories (`OptimizationHistory` or `MapOptimizationResult`),
# use [`convergenceplot`](@ref). Here `res_best` is the selected component history
# from [Banana: Adaptive Transport Map from Samples](../Examples/banana_adaptive.md):
#
# ```julia
# fig, ax, plt = convergenceplot(res_best; show_test=true)
# axislegend(ax)
# save("convergence.pdf", fig)
# ```
#
# ![Training and test objectives across adaptive iterations](../Examples/objectives.svg)
#
# The increase in the test objective at the final iteration indicates overfitting;
# the linked example explains how the map is selected using cross-validation.
#
# For a completed `OptimizationResult`, use `objectiveplot(result)` to plot the
# objective **per component**, rather than per iteration. For a vector of adaptive
# component histories, call `convergenceplot!` on a separate axis for each component.
#
# Test curves default to hidden. Enable `show_test=true` only when a test/validation
# set was used: some sample-based optimizers record zeros when no test set is present.
# Empty test vectors and nonfinite test values produce no test points.
#
# Recipes support Makie updates, for example `CairoMakie.Makie.update!(plt; dims=(2, 3))`
# on a sample plot. Mutating a matrix or map in place does not by itself notify Makie;
# pass updated inputs explicitly (preferably new arrays/map copies).

# !!! note "API"
#     See the [plotting API reference](@ref Plotting) for all plotting functions and their arguments.
