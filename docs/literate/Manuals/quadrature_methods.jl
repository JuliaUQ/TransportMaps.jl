
# # Quadrature Methods
#
# A crucial part of transport-map applications is selecting a suitable quadrature method.
# Quadrature is used in map optimization, in particular when evaluating the Kullback–Leibler divergence
#
# ```math
# \mathcal{D}_{\mathrm{KL}}\left(T_{\#} \rho \| \pi\right)=\int \Big[\log \rho(\boldsymbol{z})-\log \pi(T(\boldsymbol{a}, \boldsymbol{z}))-\log |\operatorname{det} \nabla T(\boldsymbol{a}, \boldsymbol{z})| \Big] \rho(\boldsymbol{z}) \ \mathrm{d} \boldsymbol{z}
# ```
#
# Here, $\boldsymbol{z}$ denotes the variable in the reference space with density
# $\rho(\boldsymbol{z})$, $\pi(\boldsymbol{x})$ is the target density, and
# $T(\boldsymbol{a}, \boldsymbol{z})$ is the transport map parameterized by $\boldsymbol{a}$.
#
# In practice we approximate the integral by a quadrature sum:
# ```math
# \sum_{i=1}^{N} w_{q,i}\Big[-\log\pi\bigl(T(\boldsymbol{a},\boldsymbol{z}_{q,i})\bigr)-\log |\det\nabla T(\boldsymbol{a},\boldsymbol{z}_{q,i}) |\Big]
# ```
#
# The quadrature points $\boldsymbol{z}_{q,i}$ and weights $w_{q,i}$ must be chosen so the
# sum approximates expectations with respect to the reference measure $\rho(\boldsymbol{z})$.

# Especially in Bayesian inference, where evaluating the target density $\pi(\boldsymbol{x})$
# can be expensive, using efficient quadrature methods is important to reduce the number of target evaluations [grashorn2024](@cite).

# ## Reference Density and Quadrature Choice
#
# Since quadrature is performed in the reference space $\boldsymbol{Z}$, the quadrature
# method must match said space and reference density.
# An overview is provided in the table below.

# | Quadrature rule | Reference | Notes |
# | --- | --- | --- |
# | [`MonteCarloWeights`](@ref) | any (default: $\mathcal{N}(0, 1)$) | Uses random sampling from the default reference |
# | [`LatinHypercubeWeights`](@ref) | any (default: $\mathcal{N}(0, 1)$)| Quasi-random sampling from the default reference |
# | [`GaussHermiteWeights`](@ref) | $\mathcal{N}(0, 1)$ | Gaussian-based deterministic quadrature |
# | [`SparseSmolyakWeights`](@ref) | $\mathcal{N}(0, 1)$| Sparse Gaussian rule; fewer points than full tensor product |
# | [`GaussLegendreWeights`](@ref) | $\mathcal{U}(-1, 1)$ | Uniform-based deterministic quadrature. Can also be used with $\mathcal{U}(0, 1)$ .|

# Example:
# ```julia
# map_u = PolynomialMap(2, 3, Uniform(), Softplus(), LegendreBasis())
# q_gl  = GaussLegendreWeights(5, map_u)      # uniform-aware deterministic
# q_mc  = MonteCarloWeights(1000, map_u)      # MC samples from map reference
# q_lhs = LatinHypercubeWeights(1000, map_u)  # LHS samples from map reference
# ```
# ## Monte Carlo
#
# Monte Carlo estimates an expectation by sampling from the reference density
# (default `Normal()` for `MonteCarloWeights(n, d)`) and forming the sample average.
# It is simple and robust; weights are uniform.
# Convergence is stochastic ($O(n^{-1/2})$) but independent of dimension.

#md using TransportMaps # hide
#md using Distributions # hide
#md using Plots # hide

#md using Random # hide
#md Random.seed!(42) # hide
mc = MonteCarloWeights(500, 2)
p_mc = scatter(mc.points[:, 1], mc.points[:, 2], ms=3,
    label="MC samples", title="Monte Carlo (500 pts)", aspect_ratio=1)
#md savefig("quadrature_mc.svg"); nothing # hide
# ![Monte Carlo samples](quadrature_mc.svg)

# ## Latin Hypercube Sampling (LHS)
#
# Latin Hypercube is a stratified sampling design that improves space-filling
# over plain Monte Carlo; weights remain uniform. It often reduces variance
# for low to moderate dimensions.
lhs = LatinHypercubeWeights(500, 2)
p_lhs = scatter(lhs.points[:, 1], lhs.points[:, 2], ms=3,
    label="LHS samples", title="Latin Hypercube (500 pts)", aspect_ratio=1)
#md savefig("quadrature_lhs.svg"); nothing # hide
# ![Latin Hypercube samples](quadrature_lhs.svg)

# ## Tensor-product Gauss–Hermite
#
# Gauss–Hermite quadrature provides nodes $\boldsymbol{z}_{q, i}$ and weights $w_{q, i}$ such that for
# smooth integrands the integral with respect to the Gaussian density is
# approximated very accurately. Tensor-product rules take the 1D rule in each
# coordinate and form all combinations, producing $N=n^d$ nodes for $n$ points per
# dimension.
#
hermite = GaussHermiteWeights(5, 2)
p_hermite = scatter(hermite.points[:, 1], hermite.points[:, 2],
    ms=4, label="Gauss–Hermite", title="Tensor Gauss–Hermite (5 × 5)", aspect_ratio=1)
#md savefig("quadrature_hermite.svg"); nothing # hide
# ![Gauss-Hermite tensor product sample](quadrature_hermite.svg)

# ## Sparse Smolyak Gauss–Hermite
#
# Sparse Smolyak grids combine 1D quadrature rules across dimensions with
# carefully chosen coefficients to cancel redundant high-order interactions.
# The result is substantially fewer nodes than the full tensor product while
# retaining high accuracy for mixed smooth functions. Note that Smolyak
# quadrature can produce negative weights; this is expected for higher-order
# correction terms.
#
sparse = SparseSmolyakWeights(2, 2)
p_sparse = scatter(sparse.points[:, 1], sparse.points[:, 2], ms=6,
    label="Smolyak", title="Sparse Smolyak (level 2)", aspect_ratio=1)
#md savefig("quadrature_smolyak.svg"); nothing # hide
# ![Sparse Smolyak sample](quadrature_smolyak.svg)

# ## Tensor-product Gauss-Legendre
# Gauss-Legendre quadrature integrates polynomials on bounded intervals accurately and is
# therefore well matched to reference distributions with support on $[-1, 1]$, or after
# after an affine rescaling, on $[0, 1]$.

legendre = GaussLegendreWeights(5, 2)
p_legendre = scatter(legendre.points[:, 1], legendre.points[:, 2],
    ms=4, label="Gauss–Legendre", title="Tensor Gauss–Legendre (5 × 5) on [-1, 1]", aspect_ratio=1)
#md savefig("quadrature_legendre.svg"); nothing # hide
# ![Gauss-Legendre tensor product sample](quadrature_legendre.svg)

legendre_scaled = GaussLegendreWeights(5, 2, Uniform(0,1))
p_legendre = scatter(legendre_scaled.points[:, 1], legendre_scaled.points[:, 2],
    ms=4, label="Gauss–Legendre", title="Tensor Gauss–Legendre (5 × 5) on [0, 1]", aspect_ratio=1)
#md savefig("quadrature_legendre_scaled.svg"); nothing # hide
# ![Gauss-Legendre tensor product scaled sample](quadrature_legendre_scaled.svg)
