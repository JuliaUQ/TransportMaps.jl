# # Plotting with Makie
#
# Install a [Makie](https://docs.makie.org/stable/) backend to enable plotting:
#
# ```julia
# using Pkg
# Pkg.add(["CairoMakie", "LaTeXStrings"])
# ```
#
# Loading the backend activates `TransportMapsMakieExt`. These examples use
# [CairoMakie](https://docs.makie.org/dev/explanations/backends/cairomakie).

# The shared [`transportmap_theme`](@ref) combines the
# [ggplot2](https://docs.makie.org/dev/explanations/theming/predefined_themes#theme_ggplot2)
# styling with LaTeX fonts.

#
# ## Background, foreground, and data colors
#
# The figures on this site automatically follow the documentation's light or dark
# mode. Light mode uses the original ggplot2 style with gray axis panels and dark
# text; dark mode uses a transparent background and white text and axes.
# Theme and export setup is hidden here.

using TransportMaps, CairoMakie, LaTeXStrings, Distributions
set_theme!(transportmap_theme()) # hide
#md using Main.DocPlots: save_docplot # hide

# `backgroundcolor` applies to the figure, axes, 3D panels, and legend backgrounds.
# `foregroundcolor` controls text, tick marks, axis spines, and reference curves;
# grid lines use the same color with reduced opacity. SVG and PNG exports preserve
# transparency.
#
# Set either color to any Makie color, such as `:black`, `:white`, or `"#202530"`.
# Omit both keywords to retain the original ggplot2 colors used in the demo scripts.
#
# Set data colors with each plot's `color` keyword, independently of the foreground.
# A tuple such as `color=(:steelblue, 0.5)` adds opacity. For a custom reference line
# that should follow the foreground, use `lines!(ax, x, y; color=theme(:textcolor))`.
# Themes affect figures created afterwards; recreate existing figures to apply them.
#
# ## Samples and transformations

# We set up a simple example using the example from [Banana: Map from Density](@ref):
target = MapTargetDensity(x -> logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)) #hide
M = PolynomialMap(2, 2) #hide
optimize!(M, target, SparseSmolyakWeights(3, M)) #hide
X = randn(1_500, 2) #hide
#md nothing #hide

# [`sampleplot`](@ref) scatters an `N × d` sample matrix, with one sample per row.
# Select two coordinates with `dims`.
#md save_docplot("plotting-sampleplot.svg") do # hide
fig = sampleplot(
    X; dims = (1, 2), color = (:steelblue, 1.0),
    axis = (; xlabel = L"z_1", ylabel = L"z_2"),
    figure = (; size = (400, 400))
)
#md     fig # hide
#md end # hide
# ![Samples](plotting-sampleplot.svg)

# [`transportplot`](@ref) applies a map before plotting the samples:
#md save_docplot("plotting-transform.svg") do # hide
fig = transportplot(
    M, X; dims = (1, 2), direction = :forward, color = (:purple, 1.0),
    axis = (; xlabel = L"x_1", ylabel = L"x_2"),
    figure = (; size = (400, 400))
)
#md     fig # hide
#md end # hide
# ![Samples after the transform](plotting-transform.svg)

# `direction=:forward` calls `evaluate`; `:inverse` calls `inverse`.
# For sample-fitted maps, forward evaluation maps target samples to reference space.
#
# The functions without `!` do not create a figure and axis, see [`sampleplot!`](@ref)
# and [`transportplot!`](@ref)
#
# ## Visualizing the mapping itself
#
# [`mappingplot`](@ref) draws a 1D function curve or a transformed 2D grid.
# As above, `direction` selects `evaluate` or `inverse`.
#
# ### One dimension: CDF and inverse CDF
#
# See [CDF Estimation with a 1D Transport Map](@ref) for a comparison of logistic-to-uniform
# and logistic-to-normal maps, their inverses, and their pullback densities.
# `mappingplot!` uses `direction=:inverse` to map the target into reference space.
# Dashed curves below show the exact maps.
#
# ![Logistic maps with uniform and normal references in both directions](../Examples/logistic-maps.svg)
#
# ### Two dimensions: transformed grid
#
# Pass `(xgrid, ygrid)` to set the curve resolution and `gridlines` to set the
# number of lines per direction. Using `normal_map` from [Banana: Map from Density](@ref):
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
# For the inverse, use `direction=:inverse` and a grid in the map's output space.
# Grid plots support 2D maps, not higher-dimensional slices.
#
# ## Reference and target comparison
#
# [`reference_target_plot`](@ref) compares reference and target samples for a 2D map.
# Supply reference samples for density-fitted maps, or target samples for sample-fitted
# maps. Contours show the fitted density; use `density=target` for a known density
# or `density=nothing` to hide them.
#
# Using the map and samples from  [Banana: Map from Density](@ref):
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
# For the corresponding sample-based construction, see [Banana: Map from Samples](@ref).
#
# Override the input space with `input_space=:reference` or `:target`.
# Customize the panels through `reference_axis` and `target_axis`.
#
# ## Reference-space diagnostics in higher dimensions
#
# [`referenceplot`](@ref) compares marginals and quantiles with the reference
# and shows Pearson correlations between coordinates, in any dimension.
#
# Here `fitted_map` is a sample-fitted four-dimensional map with nonlinear
# dependencies, trained on 600 synthetic samples. `validation_samples` contains
# 1,000 independent target samples. No target density is needed for these plots.

function diagnostic_target_samples(n) # hide
    U = randn(n, 4) # hide
    return hcat( # hide
        U[:, 1], U[:, 2] .+ 0.4 .* U[:, 1] .^ 2, # hide
        U[:, 3] .+ 0.6 .* U[:, 2], U[:, 4] .+ 0.5 .* U[:, 1] .* U[:, 3] # hide
    ) # hide
end # hide

training_samples = diagnostic_target_samples(600) # hide
validation_samples = diagnostic_target_samples(1000) # hide
linear_map = LinearMap(training_samples) # hide
polynomial_map = PolynomialMap(4, 2, Normal(), Softplus(), HermiteBasis()) # hide
optimize!(polynomial_map, training_samples, linear_map) # hide
fitted_map = ComposedMap(linear_map, polynomial_map) # hide
#md nothing # hide

# Pass target samples directly: the helper chooses `evaluate` or `inverse`
# and uses the map's configured reference distribution.

#md save_docplot("plotting-reference-marginals.svg") do # hide
marginal_fig = referenceplot(fitted_map, validation_samples; kind = :marginals)
#md     marginal_fig # hide
#md end # hide
# ![Four mapped validation marginals overlaid with the standard-normal reference PDF](plotting-reference-marginals.svg)
#
# In Q–Q plots, agreement follows the dashed diagonal; deviations reveal central
# or tail differences. Pass already-mapped samples to avoid repeated transformation:

Z = evaluate(fitted_map, validation_samples)
#md save_docplot("plotting-reference-qq.svg") do # hide
qq_fig = referenceplot(Z; reference = Normal(), kind = :qq)
#md     qq_fig # hide
#md end # hide
# ![Reference versus empirical quantiles for four mapped validation coordinates](plotting-reference-qq.svg)
#
# Correlations use a fixed ``[-1,1]`` color scale; gray indicates undefined
# correlations for constant coordinates.

#md save_docplot("plotting-reference-correlation.svg") do # hide
correlation_fig = referenceplot(Z; reference = Normal(), kind = :correlation)
#md     correlation_fig # hide
#md end # hide
# ![Pearson correlations between four mapped validation coordinates](plotting-reference-correlation.svg)
#
# For larger maps, select and reorder coordinates and control the panel layout:
#
# ```julia
# referenceplot(fitted_map, validation_samples; kind=:qq, dims=(1, 3, 4), ncols=2)
# referenceplot(Z; reference=Normal(), kind=:marginals, dims=(2, 4), bins=30)
# ```
#
# Use held-out target samples: round-tripping generated samples only checks inversion.
# Marginal agreement and zero correlation do not establish independence.
#
# ## Pairwise scatter-plot matrices
#
# [`plotmatrix`](@ref) combines diagonal histograms with pairwise scatter plots,
# inspired by [MATLAB's `plotmatrix`](https://de.mathworks.com/help/matlab/ref/plotmatrix.html).
#
# Reusing the four-dimensional validation samples above:

#md save_docplot("plotting-pairwise-target.svg") do # hide
pairwise_target_fig = plotmatrix(validation_samples; style = :corr)
#md     pairwise_target_fig # hide
#md end # hide
# ![Four-dimensional target samples with marginal histograms, pairwise scatter plots, and Pearson correlations](plotting-pairwise-target.svg)
#
# Pass a fitted map to transform samples to reference space and overlay its
# reference PDF on the diagonal:

#md save_docplot("plotting-pairwise-reference.svg") do # hide
pairwise_reference_fig = plotmatrix(fitted_map, validation_samples; style = :corr)
#md     pairwise_reference_fig # hide
#md end # hide
# ![Mapped reference samples with pairwise scatter plots and marginal reference PDF overlays](plotting-pairwise-reference.svg)
#
# Choose `style=:compact` (default) for the lower triangle, `:full` for all pairs,
# or `:corr` for correlations above the diagonal. Reference PDFs use the foreground color;
# constant-coordinate correlations are labelled `undefined`.
#
# Select coordinates with `dims`; `dimlabels` optionally replaces their labels:
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
# `color`.
#
# ## Term and coefficient contributions
#
# [`termplot`](@ref) shows signed coefficients, labelled by multi-index ``\alpha``.
# Their effect on the map also depends on basis scaling and nonlinear rectification.
#
# Use `measure=:removal_rms` for the RMS output change when each coefficient is
# set to zero, without refitting:
#
# ```math
# r_\alpha = \sqrt{\frac{1}{N}\sum_{i=1}^N
# \left[M^k(z_i)-M^k_{a_\alpha=0}(z_i)\right]^2}.
# ```
#
# Using the fitted map and reference samples from [Banana: Map from Density](@ref):
#
# ```julia
# fig, ax, plt = termplot(normal_map[2])
# fig, ax, plt = termplot(normal_map[2]; measure=:removal_rms,
#     samples=normal_ref_samples[1:300, :], figure=(; size=(600, 350)))
# ```
#
# ![Coefficients and removal effects for each term of the second banana-map component](../Examples/banana-density-term-contributions.svg)
#
# Removal scores depend on the samples and are not an additive or variance
# decomposition. The fitted component remains unchanged.
#
# Supply an `N × k` matrix in forward input coordinates: target for sample-fitted
# maps, reference for density-fitted maps. For composed maps, apply the linear map first:
#
# ```julia
# Z = evaluate(composed_map.linearmap, target_samples)
# termplot(composed_map.polynomialmap[k]; measure=:removal_rms, samples=Z[:, 1:k])
# ```
#
# ## Optimization results
#
# [`convergenceplot`](@ref) plots adaptive objectives by iteration. Using `res_best`
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
# The rising test objective in the final iteration indicates overfitting.
#
# For an `OptimizationResult`, `objectiveplot(result)` shows objectives **per component**.
# Plot separate adaptive component histories with `convergenceplot!` on separate axes.
#
# Enable `show_test=true` only when validation data was used; otherwise, stored
# zeros may be placeholders. Empty or nonfinite test values are omitted.
#
# Update recipes explicitly, e.g. `CairoMakie.Makie.update!(plt; dims=(2, 3))`.
# In-place changes to a matrix or map do not notify Makie.

# !!! note "API"
#     See the [plotting API reference](@ref Plotting) for all plotting functions and their arguments.
