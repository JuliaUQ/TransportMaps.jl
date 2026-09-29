# # Banana: Map from Density
#==

This example demonstrates how to use TransportMaps.jl to approximate
a "banana" distribution using polynomial transport maps.

The banana distribution is a common test case in transport map literature [marzouk2016](@cite),
defined as a standard normal in the first dimension and a normal distribution
centered at $x_1^2$ in the second dimension. This example showcases the effectiveness
of triangular transport maps for capturing nonlinear dependencies [baptista2023](@cite).

We will construct the transport map for two different reference distributions and
associated bases:
- standard normal reference ``\rho \sim \mathcal{N}(0, 1)``
- uniform reference ``\rho \sim \mathcal{U}(0, 1)``
==#

using TransportMaps
using Distributions
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())

#md using Random # hide
#md Random.seed!(123) # hide
#md nothing # hide

# ## Target density
#
# The banana density is defined as
#
# ```math
# \pi(x_1, x_2) = \phi(x_1)\,\phi(x_2 - x_1^2),
# ```
#
# where ``\phi`` is the standard normal density [marzouk2016](@cite). Its curved
# dependence makes it a useful test for triangular transport maps
# [baptista2023](@cite).

# We define its log-density as:

target_density(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
target = MapTargetDensity(target_density)
#md nothing #hide

# ## Standard-normal reference
#
# A degree-two map can represent the characteristic quadratic bend directly.
# Constructing the quadrature from the map ensures that it matches the configured
# reference density.

normal_map = PolynomialMap(2, 2, Normal(), Softplus())
normal_quadrature = SparseSmolyakWeights(3, normal_map)
normal_result = optimize!(normal_map, target, normal_quadrature)
println(normal_result)

#==
Next, we generate new samples in the reference density to assess the accuracy of the
fitted map. For this, we evaluate the [`variance_diagnostic`](@ref), defined as half
the variance of the log-ratio between the target density represented in reference
coordinates and the reference density:

```math
\mathcal{D}_{\mathrm{var}}(M)
= \frac{1}{2}\operatorname{Var}_{Z\sim\rho}
\left[
    \log\pi(M(Z))
    + \log\left|\det\nabla M(Z)\right|
    - \log\rho(Z)
\right].
```

And we compare this with the KL-divergence from the optimization.

==#

normal_ref_samples = randn(2000, 2)
normal_target_samples = evaluate(normal_map, normal_ref_samples)
var_diag = variance_diagnostic(normal_map, target, normal_ref_samples)

println("Normal-reference KL divergence: ", normal_result.minimum)
println("Normal-reference variance diag: ", var_diag)

# ## Uniform reference
#
# A uniform reference requires a (shifted) Legendre basis and associated quadrature
# on the domain ``[0, 1]^d`` (see: [Basis Functions](@ref) and [Quadrature Methods](@ref)).
# The higher polynomial degree also approximates the Gaussian quantile transformation from
# the bounded reference domain.

uniform_distribution = Uniform(0, 1)
uniform_map = PolynomialMap(
    2,
    5,
    uniform_distribution,
    Softplus(),
    ShiftedLegendreBasis(),
)
uniform_quadrature = SparseSmolyakWeights(3, uniform_map)
uniform_result = optimize!(uniform_map, target, uniform_quadrature)
println(uniform_result)

# Again, we generate new samples to assess the accuracy of the fitted map:

uniform_ref_samples = rand(uniform_distribution, 2000, 2)
uniform_target_samples = evaluate(uniform_map, uniform_ref_samples)
var_diag_uniform = variance_diagnostic(uniform_map, target, uniform_ref_samples)

println("Uniform-reference KL divergence: ", uniform_result.minimum)
println("Uniform-reference variance diag: ", var_diag_uniform)

# ## Target/reference comparison
#
# We use the same plot construction for both maps: reference samples on the left
# and their transported target samples over the true banana-density contours on
# the right.

# Choose the target-space grid for the density contours:
x₁ = range(-4, 4, length = 120)
x₂ = range(-3, 7, length = 120)
normal_comparison = reference_target_plot(
    normal_map, normal_ref_samples;
    density = target,
    xgrid = x₁, ygrid = x₂,
    reference_axis = (; limits = ((-4, 4), (-4, 4))),
)
#md save("banana-density-normal-reference.svg", normal_comparison); nothing # hide
# ![Banana transport from a standard-normal reference](banana-density-normal-reference.svg)

uniform_comparison = reference_target_plot(
    uniform_map, uniform_ref_samples;
    density = target,
    xgrid = x₁, ygrid = x₂,
    reference_axis = (; limits = ((0, 1), (0, 1))),
)
#md save("banana-density-uniform-reference.svg", uniform_comparison); nothing # hide
# ![Banana transport from a uniform reference](banana-density-uniform-reference.svg)

# Both maps recover the curved target geometry. The standard-normal reference is
# especially economical for this target, while the uniform-reference map shows
# that density-based construction is not restricted to a Gaussian reference.


# ## Visualizing the map
#
# A regular grid makes the nonlinear transformation visible: straight reference
# lines become curved target-space lines under the fitted map.

grid = range(-2, 2; length = 100)
mapping_fig = Figure(size = (600, 400))
reference_ax = Axis(mapping_fig[1, 1]; title = "Reference", xlabel = L"z_1", ylabel = L"z_2")
target_ax = Axis(mapping_fig[1, 2]; title = "Mapped grid", xlabel = L"x_1", ylabel = L"x_2")
identity_map = LinearMap(zeros(2), ones(2))
mappingplot!(reference_ax, identity_map, (grid, grid); gridlines = 9)
mappingplot!(target_ax, normal_map, (grid, grid); gridlines = 9)
#md save("banana-density-mapping.svg", mapping_fig); nothing # hide
# ![Regular reference grid and its image under the banana map](banana-density-mapping.svg)

# ## Contributions of individual terms
#
# Coefficients describe the polynomial parameterization before rectification.
# To measure their effect on the final component, we can also set each coefficient
# to zero in turn and compute the RMS output change over the same reference samples.
# These removal scores are sample-dependent, non-additive diagnostics, without refitting.

term_fig = Figure(size = (600, 350))
termplot(
    term_fig[1, 1], normal_map[2];
    axis = (; title = "Coefficients", xlabel = L"a_\alpha")
)
termplot(
    term_fig[1, 2], normal_map[2];
    measure = :removal_rms, samples = normal_ref_samples[1:300, :],
    axis = (; title = "Removal effect", xlabel = "RMS change")
)
#md save("banana-density-term-contributions.svg", term_fig); nothing # hide
# ![Signed coefficients and RMS output changes when removing each term of the second banana-map component](banana-density-term-contributions.svg)
