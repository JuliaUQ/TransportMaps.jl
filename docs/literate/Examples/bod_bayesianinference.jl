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
using Plots
using Distributions

# ### The Forward Model
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

# ### Experimental Data
#
# We have BOD measurements at five time points:

t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1e-3)
#md nothing #hide

# Let's visualize the data along with model predictions for different parameter values:

s = scatter(t, D, label="Data", xlabel="Time (t)", ylabel="Biochemical Oxygen Demand (D)",
    size=(600, 400), legend=:topleft)
## Plot model output for some parameter values
t_values = range(0, 5, length=100)
for θ₁ in [-0.5, 0, 0.5]
    for θ₂ in [-0.5, 0, 0.5]
        plot!(t_values, [forward_model(ti, [θ₁, θ₂]) for ti in t_values],
            label="(θ₁ = $θ₁, θ₂ = $θ₂)", linestyle=:dash)
    end
end
#md savefig("realizations-bod.svg"); nothing # hide
# ![BOD Realizations](realizations-bod.svg)

# ### Bayesian Inference Setup
#
# We define the posterior distribution using a standard normal prior on both parameters
# and a Gaussian likelihood for the observations:
# ```math
# \pi(\mathbf{y}|\boldsymbol{\theta}) = \prod_{t=1}^{5} \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(y_t - \mathcal{B}(t))^2\right)
# ```

function logposterior(θ)
    ## Calculate the likelihood
    likelihood = sum([logpdf(Normal(forward_model(t[k], θ), σ), D[k]) for k in 1:5])
    ## Calculate the prior
    prior = logpdf(Normal(), θ[1]) + logpdf(Normal(), θ[2])
    return prior + likelihood
end

target = MapTargetDensity(x -> logposterior(x))

# ### Creating and Optimizing the Transport Map
#
# We use a 2-dimensional polynomial transport map with degree 3 and Softplus rectifier:

M = PolynomialMap(2, 3, :normal, Softplus(), LinearizedHermiteBasis())

# Set up Gauss-Hermite quadrature for optimization:
quadrature = GaussHermiteWeights(3, 2)

# Optimize the map coefficients:
res = optimize!(M, target, quadrature)
println("Optimization result: ", res)

# ### Generating Posterior Samples
#
# Generate samples from the standard normal distribution and map them to the posterior:

samples_z = randn(1000, 2)

# Map the samples through our transport map:
mapped_samples = evaluate(M, samples_z)

# ### Quality Assessment
#
# Compute the variance diagnostic to assess the quality of our approximation:

var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)

# ### Visualization
#
# Plot the mapped samples along with contours of the true posterior density:

θ₁ = range(-0.5, 1.5, length=100)
θ₂ = range(-0.5, 3, length=100)

posterior_values = [pdf(target, [θ₁, θ₂]) for θ₂ in θ₂, θ₁ in θ₁]

scatter(mapped_samples[:, 1], mapped_samples[:, 2],
    label="Mapped Samples", alpha=0.5, color=1,
    xlabel="θ₁", ylabel="θ₂", title="Posterior Density and Mapped Samples")
contour!(θ₁, θ₂, posterior_values, colormap=:viridis, label="Posterior Density")
#md savefig("samples-bod.svg"); nothing # hide
# ![BOD Samples](samples-bod.svg)

# Finally, we can compute the pullback density to visualize how well our transport map approximates the posterior:
posterior_pullback = [pullback(M, [θ₁, θ₂]) for θ₂ in θ₂, θ₁ in θ₁]

contour(θ₁, θ₂, posterior_values ./ maximum(posterior_values);
    levels=5, colormap=:viridis, colorbar=false,
    label="Target", xlabel="θ₁", ylabel="θ₂")
contour!(θ₁, θ₂, posterior_pullback ./ maximum(posterior_pullback);
    levels=5, colormap=:viridis, linestyle=:dash,
    label="Pullback")
#md savefig("pullback-bod.svg"); nothing # hide
# ![BOD Pullback Density](pullback-bod.svg)

# We can also visually observe a good agreement between the true posterior and the TM approximation.

# ### Conditional Density and Samples

# We can compute conditional densities and generate conditional samples using the transport map.
# Therefore, we make use of the factorization of the structure of triangular maps given by the
# Knothe-Rosenblatt rearrangement [marzouk2016](@cite), [ramgraber2025](@cite):
# ```math
# \pi(\mathrm{x})=\underbrace{\pi\left(x_1\right)}_{T_1^{-1}\left(z_1\right)} \underbrace{\pi\left(x_2 \mid x_1\right)}_{T_2^{-1}\left(z_2 ; x_1\right)} \underbrace{\pi\left(x_3 \mid x_1, x_2\right)}_{T_3^{-1}\left(z_3 ; x_1, x_2\right)} \cdots \underbrace{\pi\left(x_K \mid x_1, \ldots, x_{K-1}\right)}_{T_K^{-1}\left(z_K ; x_1, \ldots, x_{K-1}\right)},
# ```

# This allows us to compute the conditional density $\pi(\theta_2 | \theta_1)$ and generate samples from this
# conditional distribution efficiently. We only need to invert the second component of the map to obtain
# $\theta_2$ given a fixed value of $\theta_1$ and then push forward samples from the reference distribution.

# We can use the `conditional_sample` function to generate samples from the conditional distribution.
# Therefore, we samples from the standard normal distribution for $z_2$ and push them through the conditional map.
# We use the previously generated samples for $z_2$ and fix $\theta_1$.
θ₁ = 0.
conditional_samples = conditional_sample(M, θ₁, randn(10_000))
#md nothing #hide

# Then, we compute the conditional density of $\theta_2$ given $\theta_1$ first analytically
# by integrating out $\theta_1$ from the joint posterior.
# We use numerical integration for this purpose and evaluate the conditional density on a grid.
θ_range = 0:0.01:2
int_analytical = gaussquadrature(ξ -> pdf(target, [θ₁, ξ]), 1000, -10., 10.)
posterior_conditional(θ₂) = pdf(target, [θ₁, θ₂]) / int_analytical
conditional_analytical = posterior_conditional.(θ_range)
#md nothing #hide

# Then we compute the conditional density using the transport map, which is given as:
# ```math
# \pi(\theta_2 | \theta_1) = \rho_2\left(T^2(\theta_1, \theta_2)^{-1}\right) \left|\frac{\partial T^2(\theta_1, \theta_2)^{-1}}{\partial \theta_2}\right|
# ```
conditional_mapped = conditional_density(M, θ_range, θ₁)
#md nothing #hide

# Finally, we plot the results:
histogram(conditional_samples, bins=50, normalize=:pdf, α=0.5,
    label="Conditional Samples", xlabel="θ₂", ylabel="π(θ₂ | θ₁=$θ₁)")
plot!(θ_range, conditional_analytical, lw=2, label="Analytical Conditional PDF")
plot!(θ_range, conditional_mapped, lw=2, label="TM Conditional PDF")
#md savefig("conditional-bod.svg"); nothing # hide
# ![BOD Conditional Density](conditional-bod.svg)
