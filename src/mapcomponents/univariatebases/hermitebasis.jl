"""
    HermiteBasis

Probabilist Hermite polynomial basis.
"""
struct HermiteBasis <: AbstractPolynomialBasis end

# Univariate probabilist's Hermite polynomials
@inline function hermite_polynomial(n::Int64, z::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return z
    else
        H_nm2 = 1.0
        H_nm1 = z
        for k in 2:n
            H_n = z * H_nm1 - (k - 1) * H_nm2
            H_nm2, H_nm1 = H_nm1, H_n
        end
        return H_nm1
    end
end

# Derivative of univariate Hermite polynomial
@inline function hermite_derivative(n::Int64, z::Real)
    n == 0 ? 0.0 : n * hermite_polynomial(n - 1, z)
end

"""
    basisfunction(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)

Evaluate `HermiteBasis` with degree `αᵢ` at `zᵢ`.
"""
@inline function basisfunction(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)
    return hermite_polynomial(Int(αᵢ), zᵢ)
end

"""
    basisfunction_derivative(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)

Evaluate derivative of `HermiteBasis` with degree `αᵢ` at `zᵢ`.
"""
@inline function basisfunction_derivative(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)
    return hermite_derivative(Int(αᵢ), zᵢ)
end

function Base.show(io::IO, ::HermiteBasis)
    print(io, "HermiteBasis()")
end

support(basis::HermiteBasis) = RealInterval(-Inf, Inf)
