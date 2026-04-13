# # Basis Functions
#
# In TransportMaps.jl, the basis must match the reference distribution used by
# the map. The reference should have the same support as the target
# distribution. If supports do not match, consider an isoprobabilistic
# transformation before constructing the map.
#
# Supported reference-basis pairings are:
#
# | Reference distribution | Support | Basis family |
# | --- | --- | --- |
# | $\mathcal{N}(0,1)$ | $(-\infty,\infty)$ | Hermite-based bases |
# | $\mathcal{U}(-1,1)$ | $[-1,1]$ | Legendre polynomials |
# | $\mathcal{U}(0,1)$ | $[0,1]$ | Shifted Legendre polynomials |
#
# This compatibility is enforced by the map constructor.

# ## Standard Normal Reference

# Often, a standard normal reference with the probability density function

# ```math
# \phi(z) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{z^2}{2}\right)
# ```

# is chosen when the reference measure should have an infinite support.
# In this case, a  probabilistic Hermite polynomial basis is a natural choice since they are
# orthogonal with respect to this density and satisfy the three-term recurrence

# ```math
# \operatorname{He}_{n+1}(z)=z \operatorname{He}_n(z)-n \operatorname{He}_{n-1}(z),
# ```

# with $\operatorname{He}_0(z)=1$ and $\operatorname{He}_1(z)=z$.

# Orthonormal polynomial bases are useful because projections are stable and
# truncated expansions give the best $L^2$ approximation under the chosen
# reference measure. In the Gaussian case, Hermite polynomials are the natural
# choice in the Wiener-Askey scheme [xiu2002](@cite).

# We also support edge-controlled Hermite variants that improve tail behavior while keeping
# the Gaussian reference [baptista2023](@cite), [ramgraber2025](@cite):
# - Probabilistic Hermite basis
# - Linearized Hermite basis
# - Gaussian-weighted Hermite basis
# - Cubic-spline-weighted Hermite basis

# ### Probabilistic Hermite Basis
#
# We first visualize the standard probabilistic Hermite basis.

using Distributions # hide
using Plots # hide
using TransportMaps # hide

# Construct the basis with [`HermiteBasis`](@ref).
basis = HermiteBasis()
z = -3:0.01:3

p1 = plot(xlabel="z", ylabel="Basis function", title="Standard Hermite Basis")
for degree in 0:4
    plot!(p1, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
#md savefig("hermite_basis_standard.svg"); nothing # hide
# ![Standard Hermite Basis](hermite_basis_standard.svg)

# If we zoom out, we can see that the tails grow quickly for large $|z|$:
z = -7:0.1:7

p2 = plot(xlabel="z", ylabel="Basis function", title="Standard Hermite Basis")
for degree in 0:4
    plot!(p2, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
#md savefig("hermite_basis_standard_zoom.svg"); nothing # hide
# ![Standard Hermite Basis](hermite_basis_standard_zoom.svg)

# ### Linearized Hermite Basis
#
# Linearized Hermite polynomials (edge-linearized basis) were introduced in
# [baptista2023](@cite) to control growth for large $|z|$ by replacing the
# polynomial with a tangent line outside data-dependent bounds $z^l,z^u$:
#
# ```math
# \mathcal{H}^{\mathrm{Lin}}_j(z)=\frac{1}{\sqrt{Z_{\alpha_j}}}
# \begin{cases}
# \mathrm{He}_j(z^l)+\mathrm{He}'_j(z^l)(z-z^l), & z< z^l \\
# \mathrm{He}_j(z), & z^l\le z \le z^u \\
# \mathrm{He}_j(z^u)+\mathrm{He}'_j(z^u)(z-z^u), & z> z^u
# \end{cases}
# ```
#
# The bounds are chosen here as the 0.01 and 0.99 quantiles of the reference.
# The normalization follows [baptista2023](@cite):
# $Z_{\alpha_j}=\alpha_j!$ for $j<k$ and $Z_{\alpha_k}=(\alpha_k+1)!$.
basis = LinearizedHermiteBasis(Normal(), 4, 1)
println("Linearization bounds: ", basis.linearizationbounds)

p3 = plot(xlabel="z", ylabel="Basis function", title="Linearized Hermite Basis")
for degree in 0:4
    plot!(p3, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
#md savefig("hermite_basis_linearized.svg"); nothing # hide
# ![Linearized Hermite Basis](hermite_basis_linearized.svg)

# ### Edge-Controlled (Weighted) Hermite Basis: Gaussian Weight
#
# Edge control modifies each Hermite polynomial with a decaying weight to reduce
# growth in the tails [ramgraber2025](@cite). Using a Gaussian weight gives:
#
# ```math
# \mathcal{H}_j^{\text{Gauss}}(z)=\mathrm{He}_j(z)\exp\left(-\tfrac{z^2}{4}\right).
# ```
basis = GaussianWeightedHermiteBasis()

p4 = plot(xlabel="z", ylabel="Basis function", title="Gaussian-Weighted Hermite Basis")
for degree in 0:4
    plot!(p4, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
#md savefig("hermite_basis_gaussian.svg"); nothing # hide
# ![Gaussian Weighted Hermite Basis](hermite_basis_gaussian.svg)

# !!! note "Note"
#     In order to preserve some extrapolation properties, the weights are only applied to polynomials of degree $j \geq 2$, as noted in [ramgraber2025](@cite).

# ### Edge-Controlled Hermite Basis: Cubic Spline Weight
#
# A cubic spline weight smoothly damps the polynomials outside a radius $r$
# [ramgraber2025](@cite). Here, $r$ is based on 0.01 and 0.99 quantiles
# $z^l, z^u$.
#
# ```math
# \mathcal{H}_j^{\mathrm{Cub}}(z)=\operatorname{He}_j(z)\left(2 u^3-3 u^2+1\right),\qquad u=\min\!\left(1,\frac{|z|}{r}\right),\; r=2\max(|z^l|,|z^u|).
# ```
basis = CubicSplineHermiteBasis(Normal())

p5 = plot(xlabel="z", ylabel="Basis function", title="Cubic Spline Weighted Hermite Basis")
for degree in 0:4
    plot!(p5, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
#md savefig("hermite_basis_cubic.svg"); nothing # hide
# ![Cubic Spline Weighted Hermite Basis](hermite_basis_cubic.svg)

# ## Uniform[-1, 1] Reference

# For a uniform reference on $[-1, 1]$, the matching family is Legendre.
# The Legendre polynomials are orthogonal with respect to the uniform measure on
# $[-1, 1]$ and satisfy

# ```math
# (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x),
# ```

# with $P_0(x)=1$ and $P_1(x)=x$.

# Use this basis with `Uniform(-1, 1)`.
basis = LegendreBasis()

z = -1:0.01:1
p = plot(xlabel="z", ylabel="Basis function", title="Legendre Basis on [-1, 1]")
for degree in 0:4
    plot!(p, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
#md savefig("legendre.svg"); nothing # hide
# ![Legendre Basis](legendre.svg)

# ## Uniform[0, 1] Reference

# For a uniform reference on $[0, 1]$, the matching family is shifted Legendre.
# These polynomials are obtained from Legendre polynomials via

# ```math
# P_n^*(x) = P_n(2x-1).
# ```

# Use this basis with `Uniform(0, 1)`.
basis = ShiftedLegendreBasis()

z = 0:0.01:1
p = plot(xlabel="z", ylabel="Basis function", title="Shifted Legendre Basis on [0, 1]")
for degree in 0:4
    plot!(p, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
#md savefig("shifted_legendre.svg"); nothing # hide
# ![Shifted Legendre Basis](shifted_legendre.svg)
