
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
# | [`TensorProductWeights`](@ref) | any | Quadrature by full tensor-product of one-dimensional quadrature rule|
# | [`GaussHermiteWeights`](@ref) | $\mathcal{N}(0, 1)$ | Tensor-product Gauss-Hermite quadrature |
# | [`GaussLegendreWeights`](@ref) | $\mathcal{U}(-1, 1)$ (or: $\mathcal{U}(0, 1)$) | Tensor-product Gauss-Legendre quadrature|
# | [`SparseSmolyakWeights`](@ref) | any (default: $\mathcal{N}(0, 1)$)| Sparse grid quadrature rule; fewer points than full tensor product |


# ## Sampling-based Quadrature
# ### Monte Carlo
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
scatter(mc.points[:, 1], mc.points[:, 2], ms=3,
    label="MC samples", title="Monte Carlo (500 pts)", aspect_ratio=1)
#md savefig("quadrature_mc.svg"); nothing # hide
# ![Monte Carlo samples](quadrature_mc.svg)

# ### Latin Hypercube Sampling
#
# Latin Hypercube is a stratified sampling design that improves space-filling
# over plain Monte Carlo; weights remain uniform. It often reduces variance
# for low to moderate dimensions.
lhs = LatinHypercubeWeights(500, 2)
scatter(lhs.points[:, 1], lhs.points[:, 2], ms=3,
    label="LHS samples", title="Latin Hypercube (500 pts)", aspect_ratio=1)
#md savefig("quadrature_lhs.svg"); nothing # hide
# ![Latin Hypercube samples](quadrature_lhs.svg)

# ## Tensor-Product Quadrature

# Tensor-product quadrature constructs multidimensional rules by combining one-dimensional
# quadrature rules across each coordinate. Given a 1D rule with ``n`` points, the tensor
# product forms all ``N = n ^d`` combinations of points in dimension ``d``.
# The number of points per dimension is given as ``n = 2^l + 1`` with the level ``l``.

# General tensor-product quadrature is performed with [`TensorProductWeights`](@ref):

# ```julia
# quad = TensorProductWeights(level, dim, knots)
# ```

# Different knot choices are available: [`GaussHermiteKnots()`](@ref), [`GaussLegendreKnots()`](@ref),
# [`ClenshawCurtisKnots()`](@ref)

# ### Tensor-product Gauss–Hermite

# Gauss–Hermite quadrature is optimized for Gaussian reference measures on ``(-\infty, \infty)``
#  and achieves spectral accuracy for smooth integrands.

hermite = GaussHermiteWeights(3, 2)
## alias: TensorProductWeights(3, 2, GaussHermiteKnots())
scatter(hermite.points[:, 1], hermite.points[:, 2],
    ms=4, label="Gauss–Hermite", title="Tensor Gauss–Hermite (level 3)", aspect_ratio=1)
#md savefig("quadrature_hermite.svg"); nothing # hide
# ![Gauss-Hermite tensor product sample](quadrature_hermite.svg)

# ### Tensor-product Gauss-Legendre

# Gauss–Legendre quadrature is optimized for bounded uniform reference measures on ``[-1, 1]``
# ``[0,1]`` after rescaling).

legendre = GaussLegendreWeights(3, 2)
## alias: TensorProductWeights(3, 2, GaussLegendreKnots())
scatter(legendre.points[:, 1], legendre.points[:, 2],
    ms=4, label="Gauss–Legendre", title="Tensor Gauss–Legendre (level 3) on [-1, 1]", aspect_ratio=1)
#md savefig("quadrature_legendre.svg"); nothing # hide
# ![Gauss-Legendre tensor product sample](quadrature_legendre.svg)

legendre_scaled = GaussLegendreWeights(3, 2, [0, 1])
scatter(legendre_scaled.points[:, 1], legendre_scaled.points[:, 2],
    ms=4, label="Gauss–Legendre", title="Tensor Gauss–Legendre (level 3) on [0, 1]", aspect_ratio=1)
#md savefig("quadrature_legendre_scaled.svg"); nothing # hide
# ![Gauss-Legendre tensor product scaled sample](quadrature_legendre_scaled.svg)

# ## Sparse-Grid Quadrature
#
# Sparse-grid methods reduce the number of quadrature points compared to full tensor products
# while maintaining accuracy for smooth functions.
#
# Sparse Smolyak grids use carefully chosen coefficients to cancel redundant high-dimensional
# interactions, reducing points from $O(n^d)$ to approximately $O(n \log(n)^{d-1})$.
# Negative weights are expected and necessary for high-order correction terms.

# Different knot choices ([`GaussHermiteKnots()`](@ref), [`GaussLegendreKnots()`](@ref),
# [`ClenshawCurtisKnots()`](@ref)) optimize sparse grids for different reference measures,
# similar to their tensor-product counterparts:

sparse = SparseSmolyakWeights(3, 2) # knots = GaussHermiteKnots()
scatter(sparse.points[:, 1], sparse.points[:, 2], ms=6,
    label="Smolyak", title="Sparse Smolyak GaussHermiteKnots", aspect_ratio=1)
#md savefig("quadrature_smolyak.svg"); nothing # hide
# ![Sparse Smolyak sample](quadrature_smolyak.svg)

sparse_legendre = SparseSmolyakWeights(3, 2, GaussLegendreKnots())
scatter(sparse_legendre.points[:, 1], sparse_legendre.points[:, 2], ms=6,
    label="Smolyak", title="Sparse Smolyak GaussLegendreKnots", aspect_ratio=1)
#md savefig("quadrature_smolyak_legendre.svg"); nothing # hide
# ![Sparse Smolyak sample](quadrature_smolyak_legendre.svg)

sparse_cc= SparseSmolyakWeights(3, 2, ClenshawCurtisKnots())
scatter(sparse_cc.points[:, 1], sparse_cc.points[:, 2], ms=6,
    label="Smolyak", title="Sparse Smolyak ClenshawCurtisKnots", aspect_ratio=1)
#md savefig("quadrature_smolyak_cc.svg"); nothing # hide
# ![Sparse Smolyak sample](quadrature_smolyak_cc.svg)
