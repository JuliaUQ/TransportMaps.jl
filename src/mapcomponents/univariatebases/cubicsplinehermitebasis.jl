"""
    CubicSplineHermiteBasis

Probabilist Hermite polynomial basis with cubic spline edge control.

# Fields
- `radius::Float64`: radius of the spline for edge control.

# Constructors
- `CubicSplineHermiteBasis(radius::Float64=3.0)`: Explicit constructor with `radius=3.0`
- `CubicSplineHermiteBasis(samples::Vector{<:Real})`: Construct radius from from 1st and 99th percentile of samples.
- `CubicSplineHermiteBasis(density::Distributions.UnivariateDistribution)`: Construct radius from 1st and 99th percentile of reference density.
"""
struct CubicSplineHermiteBasis <: AbstractPolynomialBasis
    radius::Float64

    function CubicSplineHermiteBasis(radius::Float64=3.0)
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
    basisfunction(basis::CubicSplineHermiteBasis, αᵢ::Real, zᵢ::Real)

Evaluate `CubicSplineHermiteBasis` with degree `αᵢ` at `zᵢ`.
"""
function basisfunction(basis::CubicSplineHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    r = basis.radius
    if n <= 1
        return hermite_polynomial(n, zᵢ)
    else
        return hermite_polynomial(n, zᵢ) * _cubic_weight(zᵢ, r)
    end
end

"""
    basisfunction_derivative(basis::CubicSplineHermiteBasis, αᵢ::Real, zᵢ::Real)

Evaluate derivative of `CubicSplineHermiteBasis` with degree `αᵢ` at `zᵢ`.
"""
function basisfunction_derivative(basis::CubicSplineHermiteBasis, αᵢ::Real, zᵢ::Real)
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
end

support(basis::CubicSplineHermiteBasis) = RealInterval(-Inf, Inf)
