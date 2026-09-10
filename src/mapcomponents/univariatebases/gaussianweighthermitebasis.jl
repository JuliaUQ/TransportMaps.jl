"""
    GaussianWeightedHermiteBasis

Probabilists' Hermite polynomial basis with Gaussian edge control. For degrees
``n\\geq2``, the basis function is

```math
\\psi_n(z)=H_n(z)\\exp\\!\\left(-\\frac{z^2}{4}\\right).
```

Degrees zero and one remain unweighted.
"""
struct GaussianWeightedHermiteBasis <: AbstractHermiteBasis end

function _gaussian_weight_hermite(n::Int, z::Real)
    return hermite_polynomial(n, z) * exp(-0.25 * z^2)
end

function _gaussian_weight_hermite_derivative(n::Int, z::Real)
    return n / 2 * _gaussian_weight_hermite(n - 1, z) - 0.5 * _gaussian_weight_hermite(n + 1, z)
end

"""
    basisfunction(basis::GaussianWeightedHermiteBasis, αᵢ::Int, zᵢ::Real)

Evaluate the degree-``\\alpha_i`` Gaussian-weighted Hermite basis at ``z_i``.
"""
@inline function basisfunction(basis::GaussianWeightedHermiteBasis, αᵢ::Int, zᵢ::Real)
    n = Int(αᵢ)

    if n <= 1
        return hermite_polynomial(n, zᵢ)
    else
        return _gaussian_weight_hermite(n, zᵢ)
    end
end

"""
    basisfunction_derivative(basis::GaussianWeightedHermiteBasis, αᵢ::Int, zᵢ::Real)

Evaluate the derivative of the degree-``\\alpha_i`` Gaussian-weighted Hermite
basis at ``z_i``.
"""
@inline function basisfunction_derivative(basis::GaussianWeightedHermiteBasis, αᵢ::Int, zᵢ::Real)
    n = Int(αᵢ)

    if n <= 1
        return hermite_derivative(n, zᵢ)
    else
        return _gaussian_weight_hermite_derivative(n, zᵢ)
    end
end

function Base.show(io::IO, ::GaussianWeightedHermiteBasis)
    print(io, "GaussianWeightedHermiteBasis()")
    return nothing
end


support(basis::GaussianWeightedHermiteBasis) = RealInterval(-Inf, Inf)
