# # BOD: Bayesian Inference
#
# This example estimates physical parameters of a biochemical oxygen demand
# model using transport maps. The problem was presented in [sullivan2010](@cite)
# and used as a transport-map benchmark in [marzouk2016](@cite).

using TransportMaps
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())
#md using Main.DocPlots: save_docplot # hide
using Distributions
#md using Random # hide
#md Random.seed!(123) # hide
#md nothing # hide

# ## The Forward Model and Prior
#
# The noiseless biochemical oxygen demand is
# ```math
# \mathcal{B}(t;A,B)=A(1-\exp(-Bt)),\qquad
# A\sim\mathcal{U}(0.4,1.2),\quad B\sim\mathcal{U}(0.01,0.31).
# ```
# Measurements have independent Gaussian noise with variance ``10^{-3}``.
# The forward model takes the physical parameters directly:

function forward_model(t, parameters)
    A, B = parameters
    return A * (1 - exp(-B * t))
end
#md nothing #hide

# [`PriorMap`](@ref) handles the transformation from a standard-normal auxiliary
# variable ``\theta`` to the physical prior. Its inverse goes back to ``\theta``:
# ```math
# (A,B)=P(\theta),\qquad
# P_i(\theta_i)=F_i^{-1}(\Phi(\theta_i)).
# ```

prior_map = PriorMap([Uniform(0.4, 1.2), Uniform(0.01, 0.31)])

# ## Experimental Data

t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1.0e-3)
#md nothing # hide

# Plot the data with predictions for several physical parameter values:

#md save_docplot("realizations-bod.svg") do # hide
realizations_fig = Figure(size = (600, 650))
realizations_ax = Axis(realizations_fig[1, 1]; xlabel = L"Time $t$", ylabel = L"Biochemical Oxygen Demand $D$")
scatter!(realizations_ax, t, D; label = "Data", color = theme(:textcolor))
t_values = range(0, 5; length = 100)
for A in [0.6, 0.8, 1.0], B in [0.1, 0.16, 0.22]
    lines!(
        realizations_ax, t_values, [forward_model(ti, [A, B]) for ti in t_values];
        label = L"(A = %$A,\; B = %$B)", linestyle = :dash
    )
end
Legend(realizations_fig[2, 1], realizations_ax; orientation = :horizontal, nbanks = 5)
#md     realizations_fig # hide
#md end # hide
# ![BOD data and predictions in physical parameters](realizations-bod.svg)

# ## Bayesian Inference Setup
#
# Define the likelihood and the unnormalized posterior in physical coordinates:
# ```math
# \pi(A,B\mid D)\propto L(D\mid A,B)\,\pi_0(A,B).
# ```

loglikelihood(parameters) = sum(logpdf(Normal(forward_model(t[k], parameters), σ), D[k]) for k in eachindex(t))
logposterior(parameters) = loglikelihood(parameters) + logpdf(prior_map, parameters)
physical_target = MapTargetDensity(parameters -> logposterior(parameters))

# Fit in auxiliary coordinates, where the prior is standard normal:
# ```math
# \widetilde\pi(\theta\mid D)\propto L(D\mid P(\theta))\,\phi_2(\theta).
# ```
# The prior and the Jacobian of ``P`` combine into ``\phi_2``. No additional
# Jacobian term is needed in this transformed target.

target = MapTargetDensity(θ -> loglikelihood(evaluate(prior_map, θ)) + logpdf(prior_map.reference, θ))

# ## Creating and Optimizing the Transport Map
#
# Fit a degree-three polynomial correction ``M:z\mapsto\theta`` using
# Gauss–Hermite quadrature. Compose it with the fixed prior transformation
# to obtain ``C=P\circ M:z\mapsto(A,B)``. See
# [Composing Maps](@ref) for fixed transformation layers,
# evaluation order, and density calculations:

M = PolynomialMap(2, 3, Normal(), Softplus(), LinearizedHermiteBasis())
quadrature = GaussHermiteWeights(3, M)
res = optimize!(M, target, quadrature)
println("Optimization result: ", res)
posterior_map = ComposedMap(M, prior_map; forwarddirection = :target)

# ## Generating Posterior Samples
#
# The complete composition produces physical posterior samples directly:

samples_z = randn(1000, 2)
mapped_samples = evaluate(posterior_map, samples_z)

# ## Quality Assessment
#
# The correction's variance diagnostic compares the transformed target with
# its approximation. Equivalently, one can evaluate the physical target with
# the Jacobian of the complete composition.

var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)

# ## Visualization in Physical Parameters
#
# All posterior plots use ``A`` and ``B``. Contours of the unnormalized physical
# target can be compared without knowing its evidence:

A_values = range(0.4, 1.2; length = 100)
B_values = range(0.01, 0.31; length = 100)
posterior_values = [pdf(physical_target, [A, B]) for A in A_values, B in B_values]

#md save_docplot("samples-bod.svg") do # hide
samples_fig, samples_ax, plt = sampleplot(
    mapped_samples; color = (:steelblue, 0.5),
    axis = (; xlabel = L"A", ylabel = L"B"),
)
limits!(samples_ax, 0.4, 1.2, 0.01, 0.31)
contour!(
    samples_ax, A_values, B_values, posterior_values ./ maximum(posterior_values);
    levels = [0.2, 0.4, 0.6, 0.8], colormap = :viridis
)
#md     samples_fig # hide
#md end # hide
# ![BOD posterior samples and target contours in physical parameters A and B](samples-bod.svg)

# The composition's density includes both layers' Jacobians. Compare contours
# at the same fractions of each density's peak:

posterior_pullback = [pdf(posterior_map, [A, B]) for A in A_values, B in B_values]

#md save_docplot("pullback-bod.svg") do # hide
pullback_fig = Figure(size = (600, 420))
pullback_ax = Axis(pullback_fig[1, 1]; xlabel = L"A", ylabel = L"B")
levels = 0.2:0.2:0.8
contour!(
    pullback_ax, A_values, B_values, posterior_values ./ maximum(posterior_values);
    levels, color = :steelblue, label = "Target"
)
contour!(
    pullback_ax, A_values, B_values, posterior_pullback ./ maximum(posterior_pullback);
    levels, color = :orange, linestyle = :dash, label = "Transport approximation"
)
limits!(pullback_ax, 0.4, 1.2, 0.01, 0.31)
axislegend(
    pullback_ax, [LineElement(color = :steelblue), LineElement(color = :orange, linestyle = :dash)],
    ["Target", "Transport"]; position = :lb
)
#md     pullback_fig # hide
#md end # hide
# ![BOD target and transport density in physical parameters A and B](pullback-bod.svg)

# ## Conditional Density and Samples
#
# Fix a physical value ``A=0.8`` and transform it to its auxiliary coordinate.
# Because ``P`` is diagonal, conditioning on ``A`` is equivalent to conditioning
# on that ``\theta_1``. Use the polynomial map's conditional sampler, then
# transform the sampled ``\theta_2`` values back to physical ``B`` values.
# See [Conditional Densities](@ref) for the
# triangular-map construction [marzouk2016, ramgraber2025](@cite).

fixed_A = 0.8
fixed_θ₁ = inverse(prior_map, [fixed_A, mean(prior_map.marginals[2])])[1]
conditional_θ₂ = conditional_sample(M, fixed_θ₁, randn(10_000))
conditional_samples = evaluate(prior_map, hcat(fill(fixed_θ₁, length(conditional_θ₂)), conditional_θ₂))[:, 2]
#md nothing # hide

# Numerically normalize the physical posterior over the full support of ``B``
# to obtain a reference conditional density. The unknown evidence cancels.

B_range = range(0.011, 0.309; length = 300)
conditional_normalizer = gaussquadrature(B -> pdf(physical_target, [fixed_A, B]), 200, 0.01, 0.31)
reference_conditional(B) = pdf(physical_target, [fixed_A, B]) / conditional_normalizer
conditional_reference = reference_conditional.(B_range)
#md nothing # hide

# Convert the fitted conditional density using the inverse prior Jacobian:
# ```math
# \widehat\pi(B\mid A)=\widehat\pi_\theta(\theta_2\mid\theta_1)
# \left|\frac{dP_2^{-1}(B)}{dB}\right|.
# ```

B_prior_map = PriorMap([prior_map.marginals[2]])
B_points = reshape(collect(B_range), :, 1)
θ₂_range = inverse(B_prior_map, B_points)[:, 1]
conditional_mapped = conditional_density(M, θ₂_range, fixed_θ₁) .* inverse_jacobian(B_prior_map, B_points)
#md nothing # hide

#md save_docplot("conditional-bod.svg") do # hide
conditional_fig, conditional_ax, plt = hist(
    conditional_samples; bins = 50, normalization = :pdf, color = (:steelblue, 0.5),
    label = "Samples", axis = (; xlabel = L"B", ylabel = L"\pi(B \mid A = %$fixed_A)"),
)
lines!(conditional_ax, B_range, conditional_reference; linewidth = 2, color = :orange, label = "Reference")
lines!(conditional_ax, B_range, conditional_mapped; linewidth = 2, color = :seagreen, label = "Transport")
xlims!(conditional_ax, 0.01, 0.31)
axislegend(conditional_ax; position = :lt)
#md     conditional_fig # hide
#md end # hide
# ![BOD conditional samples and density of B given physical A](conditional-bod.svg)
