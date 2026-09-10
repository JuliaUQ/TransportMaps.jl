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
using Plots

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

# First, we valuate the true target pdf:
x₁ = range(-4, 4, length = 120)
x₂ = range(-3, 7, length = 120)
target_pdf = [pdf(target, [x1, x2]) for x2 in x₂, x1 in x₁]

function reference_target_plot( # hide
        reference_samples, # hide
        target_samples; # hide
        reference_title, # hide
        reference_limits, # hide
        target_title, # hide
    ) # hide
    reference_plot = scatter( # hide
        reference_samples[:, 1], # hide
        reference_samples[:, 2]; # hide
        markersize = 2, # hide
        markerstrokewidth = 0, # hide
        alpha = 0.35, # hide
        label = "Reference samples", # hide
        xlabel = "z₁", # hide
        ylabel = "z₂", # hide
        xlims = reference_limits, # hide
        ylims = reference_limits, # hide
        aspect_ratio = 1, # hide
        title = reference_title, # hide
    ) # hide

    target_plot = contour( # hide
        x₁, # hide
        x₂, # hide
        target_pdf; # hide
        levels = 8, # hide
        linewidth = 2, # hide
        color = :viridis, # hide
        colorbar = false, # hide
        label = "Target density", # hide
        xlabel = "x₁", # hide
        ylabel = "x₂", # hide
        aspect_ratio = 1, # hide
        title = target_title, # hide
    ) # hide
    scatter!( # hide
        target_plot, # hide
        target_samples[:, 1], # hide
        target_samples[:, 2]; # hide
        markersize = 2, # hide
        markerstrokewidth = 0, # hide
        alpha = 0.45, # hide
        label = "evaluate(M, z)", # hide
    ) # hide

    return plot( # hide
        reference_plot, # hide
        target_plot; # hide
        layout = (1, 2), # hide
        size = (950, 430), # hide
        margin = 4 * Plots.mm, # hide
        left_margin = 7 * Plots.mm, # hide
        bottom_margin = 6 * Plots.mm, # hide
    ) # hide
end # hide

# And then, we compare the samples with the pdf contour:
normal_comparison = reference_target_plot(
    normal_ref_samples,
    normal_target_samples;
    reference_title = "Standard-normal reference",
    reference_limits = (-4, 4),
    target_title = "Target from normal reference",
)
#md savefig(normal_comparison, "banana-density-normal-reference.svg"); nothing # hide
# ![Banana transport from a standard-normal reference](banana-density-normal-reference.svg)

uniform_comparison = reference_target_plot(
    uniform_ref_samples,
    uniform_target_samples;
    reference_title = "Uniform reference",
    reference_limits = (0, 1),
    target_title = "Target from uniform reference",
)
#md savefig(uniform_comparison, "banana-density-uniform-reference.svg"); nothing # hide
# ![Banana transport from a uniform reference](banana-density-uniform-reference.svg)

# Both maps recover the curved target geometry. The standard-normal reference is
# especially economical for this target, while the uniform-reference map shows
# that density-based construction is not restricted to a Gaussian reference.
