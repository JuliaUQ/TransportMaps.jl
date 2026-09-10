"""
    CubicSplineHermiteBasis

Probabilists' Hermite polynomial basis with cubic edge control. For degrees
``n\\geq2``,

```math
\\psi_n(z)=H_n(z)w(z),\\qquad
w(z)=2m^3-3m^2+1,\\qquad
m=\\min\\!\\left(1,\\frac{|z|}{r}\\right).
```

Degrees zero and one remain unweighted.

# Fields
- `radius::Float64`: radius of the spline for edge control.

# Constructors
- `CubicSplineHermiteBasis(radius::Float64=3.0)`: Explicit constructor with `radius=3.0`
- `CubicSplineHermiteBasis(samples::Vector{<:Real})`: Construct radius from from 1st and 99th percentile of samples.
- `CubicSplineHermiteBasis(density::Distributions.UnivariateDistribution)`: Construct radius from 1st and 99th percentile of reference density.
"""
struct CubicSplineHermiteBasis <: AbstractHermiteBasis
    radius::Float64

    function CubicSplineHermiteBasis(radius::Float64 = 3.0)
        return new(radius)
    end
end

function CubicSplineHermiteBasis(samples::Vector{<:Real})
    bounds = [quantile(samples, 0.01), quantile(samples, 0.99)]
    r = 2 * maximum(abs.(bounds))
    return CubicSplineHermiteBasis(r)
end

function CubicSplineHermiteBasis(density::Distributions.UnivariateDistribution)
    bounds = [quantile(density, 0.01), quantile(density, 0.99)]
    r = 2 * maximum(abs.(bounds))
    return CubicSplineHermiteBasis(r)
end

function _cubic_weight(z::Real, r::Float64)
    m = min(1.0, abs(z) / r)
    return 2 * m^3 - 3 * m^2 + 1
end

function _cubic_weight_derivative(z::Real, r::Float64)
    if abs(z) < r
        m = abs(z) / r
        return (6 * abs(z)^2 / r^3 - 6 * abs(z) / r^2) * sign(z)
    else
        return 0.0
    end
end

"""
    basisfunction(basis::CubicSplineHermiteBasis, αᵢ::Int, zᵢ::Real)

Evaluate the degree-``\\alpha_i`` spline-controlled Hermite basis at ``z_i``.
"""
function basisfunction(basis::CubicSplineHermiteBasis, αᵢ::Int, zᵢ::Real)
    n = Int(αᵢ)
    r = basis.radius
    if n <= 1
        return hermite_polynomial(n, zᵢ)
    else
        return hermite_polynomial(n, zᵢ) * _cubic_weight(zᵢ, r)
    end
end

"""
    basisfunction_derivative(basis::CubicSplineHermiteBasis, αᵢ::Int, zᵢ::Real)

Evaluate the derivative of the degree-``\\alpha_i`` spline-controlled Hermite
basis at ``z_i``.
"""
function basisfunction_derivative(basis::CubicSplineHermiteBasis, αᵢ::Int, zᵢ::Real)
    n = Int(αᵢ)
    r = basis.radius
    if n <= 1
        return hermite_derivative(n, zᵢ)
    else
        return n * hermite_derivative(n - 1, zᵢ) * _cubic_weight(zᵢ, r) + hermite_polynomial(n, zᵢ) * _cubic_weight_derivative(zᵢ, r)
    end
end

function Base.show(io::IO, basis::CubicSplineHermiteBasis)
    print(io, "CubicSplineHermiteBasis(radius=$(basis.radius))")
    return nothing
end

support(basis::CubicSplineHermiteBasis) = RealInterval(-Inf, Inf)
