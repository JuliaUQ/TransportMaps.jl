# # Bayesian Inference: Biochemical Oxygen Demand (BOD) Example
#
# This example demonstrates Bayesian parameter estimation for a biochemical oxygen
# demand model using transport maps. The problem comes from environmental engineering
# and was originally presented in [sullivan2010](@cite) and later used as a benchmark
# in transport map applications [marzouk2016](@cite).
#
# The model describes the evolution of biochemical oxygen demand (BOD) in a river system
# using an exponential growth model with two uncertain parameters controlling growth
# and decay rates.

using TransportMaps
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme()) # hide
#md using Main.DocPlots: save_docplot # hide
using Distributions

# ## The Forward Model
#
# The BOD model is given by:
# ```math
# \mathcal{B}(t) = A(1-\exp(-Bt))+ \varepsilon
# ```
# where the parameters $A$ and $B$ unknown material parameters and $\varepsilon \sim \mathcal{N}(0, 10^{-3})$ represents measurement noise.
#
# The parameters $A$ and $B$ follow the prior distributions
# ```math
# A \sim \mathcal{U}(0.4, 1.2), \quad B \sim \mathcal{U}(0.01, 0.31)
# ```
#
# In order to describe the input parameters in an unbounded space, we transform them into
# a space with standard normal prior distributions $p(\theta_i) \sim \mathcal{N}(0, 1)$.
# We write $A$ and $B$ as functions of the new variables $\theta_1$ and $\theta_2$ with
# the help of a density transformation with the standard normal CDF $\Phi(\theta_i)$:
# ```math
# \begin{aligned}
# A &= 0.4 + (1.2 - 0.4) \cdot \Phi(\theta_1) \\
# B &= 0.01 + (0.31 - 0.01) \cdot \Phi(\theta_2)
# \end{aligned}
# ```

#md using Random # hide
#md Random.seed!(123) # hide
#md nothing # hide

# We define the forward model as a function of $\theta$ and $t$ in Julia:

function forward_model(t, θ)
    A = 0.4 + (1.2 - 0.4) * cdf(Normal(), θ[1])
    B = 0.01 + (0.31 - 0.01) * cdf(Normal(), θ[2])
    return A * (1 - exp(-B * t))
end
#md nothing #hide

# ## Experimental Data
#
# We have BOD measurements at five time points:

t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1.0e-3)
#md nothing #hide

# Let's visualize the data along with model predictions for different parameter values:

#md save_docplot("realizations-bod.svg") do # hide
realizations_fig = Figure(size = (600, 650))
realizations_ax = Axis(realizations_fig[1, 1]; xlabel = L"Time $t$", ylabel = L"Biochemical Oxygen Demand $D$")
scatter!(realizations_ax, t, D; label = "Data", color = theme(:textcolor))
## Plot model output for some parameter values.
t_values = range(0, 5, length = 100)
for θ₁ in [-0.5, 0, 0.5]
    for θ₂ in [-0.5, 0, 0.5]
        lines!(
            realizations_ax, t_values, [forward_model(ti, [θ₁, θ₂]) for ti in t_values];
            label = L"(\theta_1 = %$θ₁,\; \theta_2 = %$θ₂)", linestyle = :dash
        )
    end
end
Legend(realizations_fig[2, 1], realizations_ax; orientation = :horizontal, nbanks = 5)
#md     realizations_fig # hide
#md end # hide
# ![BOD Realizations](realizations-bod.svg)

# ## Bayesian Inference Setup
#
# We define the posterior distribution using a standard normal prior on both parameters
# and a Gaussian likelihood for the observations. In this likelihood, ``\mathcal{B}(t)``
# denotes the noiseless model prediction:
# ```math
# \mathcal{L}(\boldsymbol{D}\mid\boldsymbol{\theta}) = \prod_{t=1}^{5} \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(D_t - \mathcal{B}(t))^2\right)
# ```

function logposterior(θ)
    ## Calculate the likelihood
    likelihood = sum([logpdf(Normal(forward_model(t[k], θ), σ), D[k]) for k in 1:5])
    ## Calculate the prior
    prior = logpdf(Normal(), θ[1]) + logpdf(Normal(), θ[2])
    return prior + likelihood
end

target = MapTargetDensity(x -> logposterior(x))

# ## Creating and Optimizing the Transport Map
#
# We use a 2-dimensional polynomial transport map with degree 3 and Softplus rectifier:

M = PolynomialMap(2, 3, :normal, Softplus(), LinearizedHermiteBasis())

# Set up Gauss-Hermite quadrature for optimization:
quadrature = GaussHermiteWeights(3, M)

# Optimize the map coefficients:
res = optimize!(M, target, quadrature)
println("Optimization result: ", res)

# ## Generating Posterior Samples
#
# Generate samples from the standard normal distribution and map them to the posterior:

samples_z = randn(1000, 2)

# Map the samples through our transport map:
mapped_samples = evaluate(M, samples_z)

# ## Quality Assessment
#
# Compute the variance diagnostic to assess the quality of our approximation:

var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)

# ## Visualization
#
# Plot the mapped samples along with contours of the true posterior density:

θ₁ = range(-0.5, 1.5, length = 100)
θ₂ = range(-0.5, 3, length = 100)

posterior_values = [pdf(target, [θ₁, θ₂]) for θ₁ in θ₁, θ₂ in θ₂]

#md save_docplot("samples-bod.svg") do # hide
samples_fig, samples_ax, plt = sampleplot(
    mapped_samples; color = (:steelblue, 0.5),
    axis = (; xlabel = L"\theta_1", ylabel = L"\theta_2"),
)
contour!(samples_ax, θ₁, θ₂, posterior_values; colormap = :viridis)
#md     samples_fig # hide
#md end # hide
# ![BOD Samples](samples-bod.svg)

# Finally, we can compute the pullback density to visualize how well our transport map approximates the posterior:
posterior_pullback = [pullback(M, [θ₁, θ₂]) for θ₁ in θ₁, θ₂ in θ₂]

#md save_docplot("pullback-bod.svg") do # hide
pullback_fig = Figure(size = (600, 420))
pullback_ax = Axis(pullback_fig[1, 1]; xlabel = L"\theta_1", ylabel = L"\theta_2")
levels = 0.2:0.2:0.8
contour!(
    pullback_ax, θ₁, θ₂, posterior_values ./ maximum(posterior_values);
    levels, color = :steelblue, label = "Target"
)
contour!(
    pullback_ax, θ₁, θ₂, posterior_pullback ./ maximum(posterior_pullback);
    levels, color = :orange, linestyle = :dash, label = "Pullback"
)
axislegend(pullback_ax)
#md     pullback_fig # hide
#md end # hide
# ![BOD Pullback Density](pullback-bod.svg)

# We can also visually observe a good agreement between the true posterior and the TM approximation.

# ## Conditional Density and Samples

# We can compute conditional densities and generate conditional samples using the transport map.
# For a product reference, the first two components give
# [marzouk2016, ramgraber2025](@cite):
# ```math
# z_1=(T^1)^{-1}(\theta_1),\qquad
# \widehat{\pi}(\theta_2\mid\theta_1)
# =\frac{\rho_2(z_2)}{\partial_2 T^2(z_1,z_2)},
# \qquad T^2(z_1,z_2)=\theta_2.
# ```
# To sample, invert the fixed ``\theta_1`` to find ``z_1``, draw ``z_2`` from the
# reference, and evaluate ``\theta_2=T^2(z_1,z_2)``. See
# [Conditional Densities](../Manuals/conditional_densities.md) for the general construction.
#
# We can use the `conditional_sample` function to generate samples from the conditional distribution.
# We draw fresh standard-normal values for $z_2$ and evaluate the conditional map.
# We fix $\theta_1=0$.
θ₁ = 0.0
conditional_samples = conditional_sample(M, θ₁, randn(10_000))
#md nothing #hide

# For a numerical reference conditional density, hold ``\theta_1`` fixed and
# integrate the unnormalized posterior over ``\theta_2``. The unknown evidence
# cancels in the ratio. The finite integration interval approximates the full real line.
θ_range = 0:0.01:2
conditional_normalizer = gaussquadrature(ξ -> pdf(target, [θ₁, ξ]), 1000, -10.0, 10.0)
reference_conditional(θ₂) = pdf(target, [θ₁, θ₂]) / conditional_normalizer
conditional_reference = reference_conditional.(θ_range)
#md nothing #hide

# Evaluate the fitted conditional density for comparison:
conditional_mapped = conditional_density(M, θ_range, θ₁)
#md nothing #hide

# Finally, we plot the results:
#md save_docplot("conditional-bod.svg") do # hide
conditional_fig, conditional_ax, plt = hist(
    conditional_samples; bins = 50, normalization = :pdf, color = (:steelblue, 0.5),
    label = "Conditional Samples", axis = (; xlabel = L"\theta_2", ylabel = L"\pi(\theta_2 \mid \theta_1 = %$θ₁)"),
)
lines!(conditional_ax, θ_range, conditional_reference; linewidth = 2, color = :orange, label = "Numerical Reference PDF")
lines!(conditional_ax, θ_range, conditional_mapped; linewidth = 2, color = :seagreen, label = "TM Conditional PDF")
axislegend(conditional_ax)
#md     conditional_fig # hide
#md end # hide
# ![BOD Conditional Density](conditional-bod.svg)
