# # Mathematical Introduction
#
# Transport maps describe a complicated probability distribution as a deterministic
# transformation of a simpler one. This provides a common framework for sampling,
# density estimation, and Bayesian inference [marzouk2016](@cite). This introduction
# adapts that framework to the notation and workflows used in TransportMaps.jl;
# [ramgraber2025](@cite) provides a more extensive tutorial.
#
# ## Reference and target distributions
#
# Let ``\rho(\boldsymbol{z})`` denote a normalized reference density and
# ``\pi(\boldsymbol{x})`` a normalized target density in ``d`` dimensions.
# The reference is easy to sample and evaluate, typically a product of independent
# standard normal densities. We seek a differentiable, invertible map
# ``T:\mathcal{D}_{Z}\to\mathcal{D}_{X}`` such that
#
# ```math
# \boldsymbol{Z}\sim\rho,
# \qquad \boldsymbol{X}=T(\boldsymbol{Z})\sim\pi.
# ```
#
# The notation ``T_{\#}\rho=\pi`` means that ``T`` *pushes forward* the reference
# measure to the target measure. Identifying these measures with their densities,
# the change-of-variables formula gives
#
# ```math
# (T_{\#}\rho)(\boldsymbol{x})
# =\rho\!\left(T^{-1}(\boldsymbol{x})\right)
#  \left|\det\nabla T^{-1}(\boldsymbol{x})\right|.
# ```
#
# Here ``\nabla T`` is the Jacobian matrix with respect to the map input.
# An exact transport satisfies ``T_{\#}\rho=\pi``; a fitted finite-dimensional map
# usually only approximates this equality. Independent reference draws transformed
# by a fitted map are independent draws from its pushforward distribution, which
# may differ from the target.
#
# In one dimension, an increasing transport can be written in terms of the reference
# and target cumulative distribution functions:
#
# ```math
# T(z)=F_{\pi}^{-1}\!\left(F_{\rho}(z)\right).
# ```
#
# For a uniform reference on ``(0,1)``, this reduces to inverse-CDF sampling,
# ``T(z)=F_{\pi}^{-1}(z)``. See also the [CDF estimation example](../Examples/cdf_estimation.md).
#
# ## Monotone triangular maps
#
# TransportMaps.jl uses lower-triangular maps, whose ``k``-th component depends only
# on the first ``k`` inputs:
#
# ```math
# T(\boldsymbol{z})=
# \begin{pmatrix}
# T^1(z_1)\\
# T^2(z_1,z_2)\\
# \vdots\\
# T^d(z_1,\ldots,z_d)
# \end{pmatrix},
# \qquad \partial_k T^k>0.
# ```
#
# This structure follows the Knothe–Rosenblatt rearrangement
# [rosenblatt1952, knothe1957](@cite), which extends the one-dimensional CDF
# construction through successive conditional distributions. For a fixed ordering
# of the variables and suitable continuous positive densities, it defines an
# increasing triangular transport [marzouk2016](@cite).
#
# Triangularity makes the Jacobian determinant inexpensive:
#
# ```math
# \det\nabla T(\boldsymbol{z})
# =\prod_{k=1}^{d}\partial_k T^k(z_1,\ldots,z_k),
# \qquad
# \log\det\nabla T(\boldsymbol{z})
# =\sum_{k=1}^{d}\log\partial_k T^k(z_1,\ldots,z_k).
# ```
#
# To invert the map at ``\boldsymbol{x}``, solve
# ``T^1(z_1)=x_1``, then ``T^2(z_1,z_2)=x_2``, and so on.
# Each step is a scalar monotone root-finding problem. Strict monotonicity gives
# uniqueness on the component's range; global invertibility additionally requires
# that the component ranges cover the desired target support.
#
# ### Polynomial parameterization and rectification
#
# Following the [map parameterization manual](map_parameterization.md), write the
# parameterized components as ``T^k(\boldsymbol{z}_{1:k};\boldsymbol{a}_k)``, with
# ``\boldsymbol{a}_k`` the coefficients of component ``k``. The integrated
# rectification construction is [baptista2023](@cite)
#
# ```math
# T^k(\boldsymbol{z}_{1:k};\boldsymbol{a}_k)
# =f_k(\boldsymbol{z}_{1:k-1},0;\boldsymbol{a}_k)
# +\int_0^{z_k}
#  g\!\left(\partial_k f_k(\boldsymbol{z}_{1:k-1},\xi;\boldsymbol{a}_k)\right)
#  \,\mathrm{d}\xi,
# ```
#
# where ``g:\mathbb{R}\to(0,\infty)`` is a positive rectifier, such as softplus.
# Consequently, ``\partial_k T^k=g(\partial_k f_k)>0``. The unrectified function
# is represented by a finite tensor-product expansion:
#
# ```math
# f_k(\boldsymbol{z}_{1:k};\boldsymbol{a}_k)
# =\sum_{\boldsymbol{\alpha}\in\mathcal{A}_k}
#  a_{k,\boldsymbol{\alpha}}\Psi_{\boldsymbol{\alpha}}(\boldsymbol{z}_{1:k}),
# \qquad
# \Psi_{\boldsymbol{\alpha}}(\boldsymbol{z}_{1:k})
# =\prod_{j=1}^{k}\psi_{\alpha_j}(z_j).
# ```
#
# The multi-index set ``\mathcal{A}_k`` controls which terms and interactions are
# included. See [Basis Functions](basis_functions.md) for the available polynomial
# and modified bases, and [Adaptive Transport Maps](adaptive_transport_map.md) for
# data-driven enrichment. We collect all component coefficients into
# ``\boldsymbol{a}`` and write ``T(\boldsymbol{a},\boldsymbol{z})`` in the objectives below,
# as in the [optimization manual](optimization.md).
#
# ## Bayesian inference from a target density
#
# In Bayesian inference, the target is a posterior over parameters
# ``\boldsymbol{\theta}``; thus ``\boldsymbol{x}=\boldsymbol{\theta}`` in the notation above.
# Given observations ``\boldsymbol{D}``, a likelihood ``\mathcal{L}``, and prior ``p``,
#
# ```math
# \widetilde{\pi}(\boldsymbol{\theta})
# =\mathcal{L}(\boldsymbol{D}\mid\boldsymbol{\theta})p(\boldsymbol{\theta}),
# \qquad
# \pi(\boldsymbol{\theta})
# =\frac{\widetilde{\pi}(\boldsymbol{\theta})}{p(\boldsymbol{D})},
# \qquad
# p(\boldsymbol{D})=\int\widetilde{\pi}(\boldsymbol{\theta})\,\mathrm{d}\boldsymbol{\theta}.
# ```
#
# The evidence ``p(\boldsymbol{D})`` is usually unknown. Map-from-density construction
# requires only the unnormalized target and minimizes the reverse KL divergence
# [marzouk2016](@cite):
#
# ```math
# \mathcal{D}_{\mathrm{KL}}(T_{\#}\rho\,\|\,\pi)
# =\mathbb{E}_{\rho}\!\left[
#  \log\rho(\boldsymbol{Z})
#  -\log\pi\!\left(T(\boldsymbol{a},\boldsymbol{Z})\right)
#  -\log\left|\det\nabla T(\boldsymbol{a},\boldsymbol{Z})\right|
# \right].
# ```
#
# Substituting ``\log\pi=\log\widetilde{\pi}-\log p(\boldsymbol{D})`` shows that the
# evidence contributes only an additive constant independent of ``\boldsymbol{a}``.
# The reference log-density is also independent of the coefficients. Dropping both
# terms yields an equivalent minimization objective:
#
# ```math
# J(\boldsymbol{a})
# =\mathbb{E}_{\rho}\!\left[
#  -\log\widetilde{\pi}\!\left(T(\boldsymbol{a},\boldsymbol{Z})\right)
#  -\log\left|\det\nabla T(\boldsymbol{a},\boldsymbol{Z})\right|
# \right].
# ```
#
# With reference quadrature nodes ``\boldsymbol{z}_{q,i}`` and weights ``w_{q,i}``,
# the numerical problem becomes
#
# ```math
# \boldsymbol{a}^{\star}_N\in\operatorname*{arg\,min}_{\boldsymbol{a}}
# \sum_{i=1}^{N}w_{q,i}\left[
#  -\log\widetilde{\pi}\!\left(T(\boldsymbol{a},\boldsymbol{z}_{q,i})\right)
#  -\log\left|\det\nabla T(\boldsymbol{a},\boldsymbol{z}_{q,i})\right|
# \right].
# ```
#
# [Quadrature Methods](quadrature_methods.md) describes deterministic, sparse-grid,
# and sampling-based approximations of this reference expectation. The
# [optimization manual](optimization.md) covers optimizers and optional coefficient
# regularization. Gradient-based methods such as the default L-BFGS use derivatives
# of the target log-density, including derivatives through a forward model when it
# appears in the likelihood. The objective need not be convex.
#
# In the package, pass a callable **log-density** ``\log\widetilde{\pi}`` to
# `MapTargetDensity`, then fit with `optimize!(M, target, quadrature)`.
# For this direction, `evaluate(M, reference_samples)` produces approximate target
# samples. No posterior training samples are required. The
# [Bayesian BOD example](../Examples/bod_bayesianinference.md) demonstrates the full
# workflow; [grashorn2024](@cite) discusses transport-based Bayesian inference in
# structural health monitoring.
#
# ## Learning a map from samples
#
# When target samples ``\{\boldsymbol{x}_i\}_{i=1}^{N}`` are available, fit the opposite
# direction, ``S:\mathcal{D}_{X}\to\mathcal{D}_{Z}``, so that ``S_{\#}\pi\approx\rho``.
# An exact pair satisfies ``S=T^{-1}``. Minimizing
# ``\mathcal{D}_{\mathrm{KL}}(S_{\#}\pi\,\|\,\rho)`` gives, after omitting a
# coefficient-independent term, the empirical objective [marzouk2016, baptista2023](@cite)
#
# ```math
# J_{\mathrm{samples}}(\boldsymbol{a})
# =-\frac{1}{N}\sum_{i=1}^{N}\left[
#  \log\rho\!\left(S(\boldsymbol{a},\boldsymbol{x}_i)\right)
#  +\log\left|\det\nabla S(\boldsymbol{a},\boldsymbol{x}_i)\right|
# \right].
# ```
#
# This is also maximum likelihood estimation of the normalized model density
#
# ```math
# \widehat{\pi}(\boldsymbol{x})
# =\rho\!\left(S(\boldsymbol{x})\right)
#  \left|\det\nabla S(\boldsymbol{x})\right|,
# ```
#
# provided ``S`` maps the target domain bijectively onto the reference domain.
# The same polynomial construction applies, now with target coordinates as inputs.
# Samples can come from observations, simulations, or an existing sampler; MCMC is
# one possible source rather than a requirement.
#
# The package stores the fitted direction: for a sample-fitted map,
# `evaluate(M, target_samples)` maps to the reference and
# `inverse(M, reference_samples)` generates approximate target samples.
# `pullback(M, x)` evaluates the model density above. See
# [Banana: Map from Samples](../Examples/banana_mapfromsamples.md), including optional
# linear standardization with `LinearMap` and `ComposedMap`.
