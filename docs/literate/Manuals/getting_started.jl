# # Getting Started with TransportMaps.jl
#
# Turn independent standard-normal draws into samples from a correlated Gaussian.
# In four steps, you will **define a target, fit a map, generate samples, and check
# the result**. For the concepts and notation, start with the [Introduction](introduction.md).
#
# ![Independent reference samples transformed into a correlated Gaussian target](samples.svg)
#
# ## Install and Load
#
# Install the package and the distribution tools used in this guide:
#
# ```julia
# using Pkg
# Pkg.add(["TransportMaps", "Distributions"])
# ```

using TransportMaps, Distributions
using Random # hide
Random.seed!(1234) # hide
#md using Main.DocPlots: save_docplot # hide

# ## 1. Define the Target
#
# Our target is a two-dimensional Gaussian with zero mean, unit marginal variances,
# and correlation ``0.8``. Wrap its log-density in [`MapTargetDensity`](@ref):

gaussian = MvNormal(zeros(2), [1.0 0.8; 0.8 1.0])
target = MapTargetDensity(x -> logpdf(gaussian, x))
#md nothing # hide

# !!! tip "Pass a log-density"
#     The callable must return the **log** of the target density. It may be
#     unnormalized, which is useful for Bayesian posteriors.

# ## 2. Fit the Map
#
# Create a two-dimensional map with polynomial degree one. Its defaults use a
# standard-normal reference, a Hermite-based expansion, and softplus rectification.
# Construct quadrature for that reference, then fit the coefficients:

M = PolynomialMap(2, 1)
quadrature = GaussHermiteWeights(2, M)
result = optimize!(M, target, quadrature)
#md nothing # hide

# [`optimize!`](@ref) updates `M` in place. The returned `result` contains the solver's
# convergence information. You can reuse the fitted map to generate new samples.

# ## 3. Generate Samples
#
# Draw from the reference and transform the draws with [`evaluate`](@ref):

reference_samples = randn(1_000, 2)
target_samples = evaluate(M, reference_samples)
#md nothing # hide

# Each row is one sample; each column is one coordinate. `target_samples` contains
# independent draws from the fitted approximation to the target.
# In the figure above, the circular reference cloud becomes a tilted target cloud;
# the contours show the known Gaussian target density.

# ## 4. Check the Result
#
# Evaluate the variance diagnostic on the reference samples:

variance_diagnostic(M, target, reference_samples)

# A value close to zero indicates little variation in the log-density mismatch at
# the sampled points. For this simple target, it should be small. This is a useful
# diagnostic, rather than a guarantee of accuracy over the entire domain.

# ## Plot Your Map
#
# The `reference_target_plot` result is shown at the top of this page. To reproduce
# it, install a Makie backend once:
#
# ```julia
# Pkg.add("CairoMakie")
# ```
#
# Then load it and call [`reference_target_plot`](@ref):

using CairoMakie
set_theme!(transportmap_theme()) # hide
#md save_docplot("samples.svg") do # hide
fig = reference_target_plot(M, reference_samples; density = target)
#md     fig # hide
#md end # hide

# [`mappingplot`](@ref) shows the transformation itself by mapping a regular grid.
# Here the grid becomes sheared, reflecting the correlation in the target:

grid = range(-2, 2; length = 100)
#md save_docplot("mapping.svg") do # hide
fig, ax, plt = mappingplot(
    M, (grid, grid); gridlines = 9,
    axis = (; xlabel = "x₁", ylabel = "x₂", aspect = DataAspect()),
    figure = (; size = (400, 500))
)
#md     fig # hide
#md end # hide
# ![Regular reference grid transformed by the fitted Gaussian map](mapping.svg)
#
# ## Where to Go Next
#
# | Your next task | Guide |
# |:--|:--|
# | Learn from observations instead of a density | [Banana: Map from Samples](../Examples/banana_mapfromsamples.md) |
# | Fit a nonlinear target density | [Banana: Map from Density](../Examples/banana_mapfromdensity.md) |
# | Infer parameters from measurements | [Bayesian inference with BOD data](../Examples/bod_bayesianinference.md) |
# | Choose bases, rectifiers, and map structure | [Basis Functions](basis_functions.md) and [Map Parameterization](map_parameterization.md) |
# | Tune integration or solver settings | [Quadrature Methods](quadrature_methods.md) and [Optimization](optimization.md) |
