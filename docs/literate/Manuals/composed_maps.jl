# # Composing Maps
#
# Composing maps allows a combination of simple transformations into a more flexible transport.
# [`ComposedMap`](@ref) chains fixed coordinate transformations and fitted maps, providing
# evaluation, inversion, and density calculations for the complete composition.
#
# For maps ``T_1,\ldots,T_K`` with compatible domains, define intermediate
# coordinates ``u_0=z`` and ``u_k=T_k(u_{k-1})``. The composition is
#
# ```math
# C=T_K\circ\cdots\circ T_1,\qquad
# C(z)=T_K(\cdots T_1(z)),\qquad
# C^{-1}=T_1^{-1}\circ\cdots\circ T_K^{-1}.
# ```
#
# Write ``\mathbf{J}_C(z)`` for the Jacobian matrix of ``C``, whose entries are
#
# ```math
# [\mathbf{J}_C(z)]_{ij}=\frac{\partial C_i}{\partial z_j}(z).
# ```
#
# The chain rule gives its log absolute determinant:
#
# ```math
# \log|\det \mathbf{J}_C(z)|=
# \sum_{k=1}^K\log|\det \mathbf{J}_{T_k}(u_{k-1})|.
# ```
#
# Each determinant is evaluated at its layer's input. Composing lower
# triangular maps with positive diagonal derivatives preserves these properties.
# For a reference-to-target map and reference density ``\rho``, the resulting
# target density satisfies
#
# ```math
# \log q(C(z))=\log\rho(z)-\log|\det \mathbf{J}_C(z)|.
# ```
#
# ## Constructing and Evaluating a Composition
#
# Constructor arguments follow **evaluation order**: the first map is applied
# first. Construct the layers separately, then combine them:

#md using TransportMaps, Distributions # hide

linear = LinearMap([1.0, -1.0], [2.0, 3.0])
correction = PolynomialMap(2, 2)
C = ComposedMap(linear, correction; forwarddirection = :reference)

x = [0.2, -0.4]
z = evaluate(C, x)          # correction(linear(x))
x_recovered = inverse(C, z) # inverse layers in reverse order
#md nothing # hide

# These constructors initialize the maps. Fit the polynomial layer separately
# using the procedures in [Optimization](@ref "Optimization of the Map Coefficients"). `ComposedMap` uses
# the supplied layers and their current parameters.
#
# Matrices contain one observation per row:

X = [0.2 -0.4; 0.5 0.1]
Z = evaluate(C, X)
X_recovered = inverse(C, Z)
#md nothing # hide

# Tuples, vectors, and nested compositions are also accepted:
# `ComposedMap((linear, correction))`, `ComposedMap([linear, correction])`, and
# `ComposedMap(ComposedMap(linear), correction)` have the same evaluation order.
# Layers must have matching dimensions and compatible supports. `LinearMap()`
# is a dimension-free identity.
#
# ## Fixed Coordinate Transformations
#
# The following maps can be used as fixed layers alongside fitted polynomial maps:
#
# | Map | Forward transformation | Typical use |
# |:--|:--|:--|
# | [`LinearMap`](@ref) | ``z_i=(x_i-\mu_i)/\sigma_i`` | Componentwise centering and scaling |
# | [`LaplaceMap`](@ref) | ``z=L^{-1}(x-\mu)`` | Whitening with a full covariance or a Laplace approximation |
# | [`PriorMap`](@ref) | ``x_i=F_i^{-1}(G(z_i))`` | Reference-to-prior transformation for independent continuous marginals |
#
# ### LinearMap and LaplaceMap
#
# `LinearMap` accepts means and standard deviations directly, as above, or
# estimates them from samples. `LaplaceMap` also accounts for correlations:

samples = [0.5 1.0; 1.5 -0.5; 2.0 0.5; 3.0 1.5; 2.5 -1.0]
standardization = LinearMap(samples)
whitening = LaplaceMap(samples)
whitened_map = ComposedMap(whitening, correction; forwarddirection = :reference)
#md nothing # hide

# `LaplaceMap(target, initial_point)` instead finds the mode of a
# `MapTargetDensity` and estimates covariance from its log-density Hessian.
# For an unnormalized target ``\bar\pi``, the Laplace approximation is
#
# ```math
# \widehat\mu=\operatorname*{arg\,max}_x\log\bar\pi(x),\qquad
# H=-\nabla_x^2\log\bar\pi(\widehat\mu),\qquad
# \pi_{\mathrm{Laplace}}(x)=\mathcal{N}(x;\widehat\mu,H^{-1}).
# ```
#
# The Hessian ``H`` must be positive definite at the fitted mode. Factoring
# ``H^{-1}=LL^{\mathsf T}`` gives the forward whitening map
# ``z=L^{-1}(x-\widehat\mu)``. The target's normalizing constant is not needed.
# For example, the following Gaussian target has mode ``(1,-0.5)`` and
# covariance ``\operatorname{diag}(1,4)``, so its Laplace approximation is exact:

target = MapTargetDensity(x -> -(x[1] - 1)^2 / 2 - (x[2] + 0.5)^2 / 8)
density_whitening = LaplaceMap(target, [0.0, 0.0])
laplace_chain = ComposedMap(density_whitening, correction; forwarddirection = :reference)
#md nothing # hide

# Both maps standardize in their forward direction; `inverse` returns to the
# original coordinates. Standardization alone need not make a non-Gaussian
# density Gaussian.
#
# ### PriorMap
#
# `PriorMap` accepts a vector of independent continuous prior marginals. For
# their CDFs ``F_i`` and reference CDF ``G``, it evaluates
# ``P_i(z_i)=F_i^{-1}(G(z_i))``. Its inverse transforms physical prior
# coordinates to reference coordinates:

prior = PriorMap([LogNormal(0.3, 0.7), Uniform(0.4, 1.2)])
physical_point = evaluate(prior, [0.0, 0.0])
reference_point = inverse(prior, physical_point)
#md nothing # hide

# The default reference is `Normal()`. Choose a different continuous univariate
# reference with the keyword; it is repeated independently in every dimension:

uniform_prior = PriorMap(
    [LogNormal(0.3, 0.7), Uniform(0.4, 1.2)];
    reference = Uniform(0, 1)
)
#md nothing # hide

# To combine a reference-space correction ``T`` with the physical prior map
# ``P``, apply the correction first, then the prior transformation:

physical_map = ComposedMap(correction, prior; forwarddirection = :target)
parameters = evaluate(physical_map, [0.0, 0.0])
#md nothing # hide

# The correction's output must lie within the prior map's reference support.
# In particular, a bounded reference requires a correction that preserves that
# support. Out-of-support inputs to `PriorMap` throw `DomainError`.
#
# In Bayesian inference, transforming a physical posterior through an exact
# prior map absorbs the prior and its Jacobian into the reference density:
#
# ```math
# \pi(x| y)\propto L(y| x)\,\pi_0(x),\qquad
# \widetilde\pi(z| y)\propto L(y| P(z))\,\rho(z).
# ```
#
# See [BOD: Bayesian Inference](@ref)
# for fitting in these coordinates and evaluating densities in physical parameters.
#
# ## Direction and Density Evaluation
#
# `forwarddirection` specifies the endpoints of the **whole** composition:
#
# - `:reference`: target to reference; sampling uses `inverse`.
# - `:target`: reference to target; sampling uses `evaluate`.
#
# The keyword does not reverse any layer. If omitted, directions are inferred
# from layer metadata and must agree; linear-only chains default to `:reference`.
#
# The reference density is taken from the last layer with reference metadata
# for `:reference`, or the first such layer for `:target`. Linear-only chains
# use a product standard normal. Set the reference explicitly when needed:

C_with_reference = ComposedMap(
    linear, correction;
    forwarddirection = :reference, reference = Normal()
)
#md nothing # hide

# `pdf` and `logpdf` always evaluate the target-density approximation at target
# coordinates. `jacobian` returns the forward determinant, and
# `logabsdetjacobian` accumulates layer log-determinants:

density = pdf(C, x)
log_density = logpdf(C, x)
determinant = jacobian(C, x)
log_determinant = logabsdetjacobian(C, x)
#md nothing # hide

# These operations also accept matrices of row-wise points. `pullback(C, x)`
# follows the package convention and is an alias for `pdf(C, x)`.
# `pushforward(C, target, z)` evaluates a supplied target density in reference
# coordinates, including the change-of-variables Jacobian.
#
# ## Adding a Fitted Correction
#
# For an existing reference-to-target map ``T``, a new reference-space
# correction ``Q`` acts before the old map: ``T_{\mathrm{new}}=T\circ Q``.
# In evaluation order, prepend it:
#
# ```julia
# T_new = ComposedMap(Q, T; forwarddirection = :target)
# ```
#
# For a target-to-reference map ``S``, append a correction ``R``:
#
# ```julia
# S_new = ComposedMap(S, R; forwarddirection = :reference)
# ```
#
# The [Beam: Sequential Inference with Composed Maps](@ref)
# example shows how to fit these corrections and visualize successive posteriors.
