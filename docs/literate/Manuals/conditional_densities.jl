# # Conditional Densities
#
# When constructing a transport map, the triangular structure of the Knothe-Rosenblatt rearrangement provides a systematic way to factorize the joint density into a product of conditional densities:
#
# ```math
# \pi(\boldsymbol{x}) = \pi(x_1) \pi(x_2 | x_1) \pi(x_3 | x_1, x_2) \cdots \pi(x_k | x_1, \dots, x_{k-1}).
# ```
#
# In Refs. [marzouk2016](@cite) and [ramgraber2025](@cite), the authors show that these conditional densities are obtained sequentially by
# inverting the map components after one another.
# We use the reference-to-target map ``T`` defined in the [Introduction](introduction.md),
# with an independent product reference ``\rho(\boldsymbol{z})=\prod_j\rho_j(z_j)``.
#
# The conditional density for $x_k$ given $x_1, \dots, x_{k-1}$ is defined as:
#
# ```math
# \widehat{\pi}(x_k \mid x_1, \dots, x_{k-1}) = \rho_k(z_k) \left| \partial_k T^k(z_1,\ldots,z_k) \right|^{-1},\qquad \boldsymbol{z}_{1:k}=T_{1:k}^{-1}(\boldsymbol{x}_{1:k}).
# ```
#
# Here ``T_{1:k}`` denotes the first ``k`` components, inverted sequentially, and
# ``\rho_k`` is the one-dimensional reference density. The hat distinguishes the
# fitted model from the true target conditional density.
#
# To sample given ``\boldsymbol{x}_{1:k-1}``, first invert those fixed coordinates
# to obtain ``\boldsymbol{z}_{1:k-1}``. Draw a fresh ``z_k\sim\rho_k``, then evaluate
# ``x_k=T^k(z_1,\ldots,z_k)``.

# ## Setting up a Transport Map
#
# For this example, we'll use the *banana* distribution as our target density,
# also used in [Banana: Map from Density](@ref).
# The banana distribution is defined as a:
#
# ```math
# \pi(x_1, x_2) = \phi(x_1) \cdot \phi(x_2 - x_1^2),
# ```
# where $\phi$ is the standard normal PDF.

# Define the target log-density and construct a map as in [Getting Started](getting_started.md):
using TransportMaps
using CairoMakie # hide
using LaTeXStrings # hide
set_theme!(transportmap_theme()) # hide
#md using Main.DocPlots: save_docplot # hide
using Distributions

logtarget(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
target = MapTargetDensity(logtarget)
#md nothing #hide

# Define the map and quadrature; and optimize the map:
M = PolynomialMap(2, 2, :normal, Softplus(), HermiteBasis())
quadrature = GaussHermiteWeights(3, M)

## Optimize the map:
optimize!(M, target, quadrature)
#md nothing #hide

# ## Conditional Density Evaluation
#
# Now we can compute conditional densities. For simplicity, we look at a two-dimensional example,
# so we are interested in the conditional density $\pi(x_2 | x_1 = 0.5)$.

# Define the conditioning variable and the variable to evaluate the conditional density:
x₁ = 0.5
x₂ = 0.8
density = conditional_density(M, x₂, x₁)
println("Conditional density π(x₂=$x₂ | x₁=$x₁) = $density")

# Evaluate conditional density for multiple values:
x₂_values = range(-3, 3, length = 100)
densities = conditional_density(M, x₂_values, x₁)
#md nothing #hide

# Plot the conditional density:
#md save_docplot("conditional-density.svg") do # hide
fig, ax, plt = lines( # hide
    x₂_values, densities; linewidth = 2, # hide
    axis = (; xlabel = L"x_2", ylabel = L"\pi(x_2 \mid x_1 = %$x₁)"), # hide
) # hide
#md     fig # hide
#md end # hide
# ![Conditional Density](conditional-density.svg)

# ## Conditional Sampling
#
# We can also sample from the conditional density $\pi(x_2 | x_1)$.
# First, we sample $z_2$ in the standard normal space and then evaluate the conditional map:

# Single value sampling:
z₂ = randn()
cond_sample = conditional_sample(M, x₁, z₂)
println("Conditional sample for z₂=$z₂: x₂=$cond_sample")

# Multiple samples:
z₂_values = randn(10_000)
cond_samples = conditional_sample(M, x₁, z₂_values)
#md nothing #hide

# Create a histogram of the conditional samples and overlay the fitted conditional density:
#md save_docplot("conditional-sampling.svg") do # hide
fig, ax, plt = hist( # hide
    cond_samples; bins = 50, normalization = :pdf, color = (:steelblue, 0.7), # hide
    label = "Conditional Samples", axis = (; xlabel = L"x_2", ylabel = "Density"), # hide
) # hide
lines!(ax, x₂_values, densities; linewidth = 2, color = :orange, label = "Conditional Density") # hide
axislegend(ax) # hide
#md     fig # hide
#md end # hide
# ![Conditional Sampling](conditional-sampling.svg)

# ## Comparison with True Conditional Density
#
# Let's compare our transport map's conditional density with the true conditional density
# of the target distribution:

# True conditional density for the banana distribution:
true_banana_conditional_density(x₂, x₁) = pdf(Normal(x₁^2, 1), x₂)
#md nothing #hide

# Compute true conditional densities:
true_densities = [true_banana_conditional_density(x₂, x₁) for x₂ in x₂_values]
#md nothing #hide

# Plot comparison:
#md save_docplot("conditional-comparison.svg") do # hide
fig, ax, plt = lines( # hide
    x₂_values, densities; linewidth = 2, label = "TM Conditional", # hide
    axis = (; xlabel = L"x_2", ylabel = L"\pi(x_2 \mid x_1 = %$x₁)"), # hide
) # hide
lines!(ax, x₂_values, true_densities; linewidth = 2, linestyle = :dash, label = "True Conditional") # hide
axislegend(ax) # hide
#md     fig # hide
#md end # hide
# ![Conditional Comparison](conditional-comparison.svg)

# ## Multiple Conditioning Scenarios
#
# Let's explore how the conditional density π(x₂ | x₁) changes as we vary x₁.
# This shows the nonlinear structure of the banana distribution:

x₁_values = [-0.6, 0.0, 1.0, 2.0]

#md save_docplot("multiple-conditioning.svg") do # hide
fig = Figure(size = (600, 600)) # hide
ax = Axis( # hide
    fig[1, 1]; xlabel = L"x_2", ylabel = L"\pi(x_2 \mid x_1)" # hide
) # hide
colors = [:steelblue, :orange, :seagreen, :purple] # hide
for (i, x₁_val) in enumerate(x₁_values) # hide
    densities_cond = conditional_density(M, x₂_values, x₁_val) # hide
    true_densities_cond = [true_banana_conditional_density(x₂, x₁_val) for x₂ in x₂_values] # hide
    lines!(ax, x₂_values, densities_cond; linewidth = 2, label = L"TM: $x_1 = %$x₁_val$", color = colors[i]) # hide
    lines!( # hide
        ax, x₂_values, true_densities_cond; linewidth = 2, linestyle = :dash, # hide
        label = L"True: $x_1 = %$x₁_val$", color = colors[i] # hide
    ) # hide
end # hide
Legend(fig[2, 1], ax; orientation = :horizontal, nbanks = 4) # hide
#md     fig # hide
#md end # hide
# ![Multiple Conditioning](multiple-conditioning.svg)

# !!! note "Note"
#     For a more comprehensive example with real-world applications,
#     see the [BOD: Bayesian Inference](@ref).
