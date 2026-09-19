# # Getting Started with TransportMaps.jl
#
# This guide will help you get started with TransportMaps.jl for constructing and using transport maps.
#
# ## Basic Concepts
#
# ### What is a Transport Map?
#
# A transport map $T: \boldsymbol{Z} \mapsto \boldsymbol{X}$ is a mapping from reference space $\boldsymbol{Z} \sim \rho(\boldsymbol{z})$ to the target space $\boldsymbol{X} \sim \pi(\boldsymbol{x})$ [marzouk2016](@cite).
# Hence, the inverse map $T^{-1}: \boldsymbol{X} \mapsto \boldsymbol{Z}$ maps from the target to the reference space.

# ### Triangular Maps
#
# TransportMaps.jl focuses on **triangular transport maps** [baptista2023](@cite),
# following the Knothe-Rosenblatt rearrangement [knothe1957](@cite).
# This structure ensures that the map is invertible and the Jacobian determinant is easy to compute.
# A triangular map in $n$ dimensions has the form:

# ```math
# T(\boldsymbol{z}) =
# \left(\begin{array}{c}
# T_1(z_1) \\
# T_2(z_1, z_2) \\
# T_3(z_1, z_2, z_3) \\
# \vdots \\
# T_n(z_1, z_2, \dots, z_n)
# \end{array}
# \right)
# ```

# The inverse map $T^{-1}$ can be computed sequentially by inverting each component.

# ## First Example: A Simple 2D Transport Map

using TransportMaps
using Distributions
using Random
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())
using LinearAlgebra

# Let's create a simple 2D transport map:

# Set random seed for reproducibility
Random.seed!(1234)
#md nothing #hide

# Create a 2D polynomial map with degree 2
M = PolynomialMap(2, 2, Normal(), Softplus())

# The map is initially identity (coefficients are zero)
println("Initial coefficients: ", getcoefficients(M))

# ### Defining a Target Distribution
#
# For optimization, you need to define your target probability density.
# Let's start with a simple correlated Gaussian:

function correlated_gaussian(x; ρ = 0.8)
    Σ = [1.0 ρ; ρ 1.0]
    return logpdf(MvNormal(zeros(2), Σ), x)
end
#md nothing #hide

# Then, we construct the `MapTargetDensity` object. In the default case, automatic differentiation is used with [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/)
# AD is implemented with [`DifferentiationInterface.jl`](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/).
# This allows for the use of other packages supported by the interface, e.g., `Mooncake.jl`, `Zygote.jl` or `FiniteDiff.jl` for finite difference approximations.
# For more information, we also refer to [dalle2025](@cite).

# Create a MapTargetDensity object for optimization
target_density = MapTargetDensity(correlated_gaussian)

# ### Setting up Quadrature
#
# Choose an appropriate quadrature scheme for map optimization:

# Gauss-Hermite quadrature (good for Gaussian-like targets)
quadrature = GaussHermiteWeights(2, 2)  # level l=2, (2^l + 1 = 5 points per dimension), 2D
## alternative options:
## quadrature = MonteCarloWeights(1000, 2)  # 1000 samples, 2D
## quadrature = LatinHypercubeWeights(1000, 2)
## quadrature = SparseSmolyakWeights(3, 2)  # Level 3, 2D


# ### Optimizing the Map
#
# Fit the transport map to your target distribution:
result = optimize!(M, target_density, quadrature)

# ### Generating Samples
#
# Once optimized, use the map to generate samples:

# Generate reference samples (standard Gaussian)
n_samples = 1000
reference_samples = randn(n_samples, 2)

# Transform to target distribution
target_samples = evaluate(M, reference_samples)

# ### Visualizing Results
#
# Let's plot both the reference, target samples and the pdf:
x₁ = range(-4, 4, length = 120)
x₂ = range(-4, 4, length = 120)

comparison = reference_target_plot(
    M, reference_samples;
    density = target_density,
    xgrid = x₁, ygrid = x₂,
)
#md save("samples.svg", comparison); nothing # hide
# ![Transport Map Samples](samples.svg)

# ### Evaluating Map Quality
#
# Check how well your map approximates the target:

# Variance diagnostic (should be close to zero for a good map)
var_diag = variance_diagnostic(M, target_density, reference_samples)
println("Variance diagnostic: ", var_diag)

# You can also check the Jacobian determinant
sample_point = [0.0, 0.0]
jac = jacobian(M, sample_point)
det_jac = det(jac)
println("Jacobian determinant at origin: ", det_jac)

# ### Working with Different Rectifiers
#
# The rectifier function affects the map's behavior. Let's compare different options:

# ShiftedELU rectifier
M_elu = PolynomialMap(2, 2, Normal(), ShiftedELU())
result_elu = optimize!(M_elu, target_density, quadrature)
var_diag_elu = variance_diagnostic(M_elu, target_density, reference_samples)

println("Variance diagnostics:")
println("  Softplus: ", var_diag)
println("  ShiftedELU: ", var_diag_elu)
