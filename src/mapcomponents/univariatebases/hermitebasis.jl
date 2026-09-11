"""
    HermiteBasis

Probabilists' Hermite polynomial basis, defined by

```math
H_0(z)=1,\\qquad H_1(z)=z,\\qquad
H_n(z)=zH_{n-1}(z)-(n-1)H_{n-2}(z).
```
"""
struct HermiteBasis <: AbstractHermiteBasis end

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
    return n == 0 ? 0.0 : n * hermite_polynomial(n - 1, z)
end

"""
    basisfunction(basis::HermiteBasis, αᵢ::Int, zᵢ::Real)

Evaluate the degree-``\\alpha_i`` Hermite polynomial ``H_{\\alpha_i}(z_i)``.
"""
@inline function basisfunction(basis::HermiteBasis, αᵢ::Int, zᵢ::Real)
    return hermite_polynomial(Int(αᵢ), zᵢ)
end

"""
    basisfunction_derivative(basis::HermiteBasis, αᵢ::Int, zᵢ::Real)

Evaluate
``H'_{\\alpha_i}(z_i)=\\alpha_i H_{\\alpha_i-1}(z_i)``.
"""
@inline function basisfunction_derivative(basis::HermiteBasis, αᵢ::Int, zᵢ::Real)
    return hermite_derivative(Int(αᵢ), zᵢ)
end

function Base.show(io::IO, ::HermiteBasis)
    print(io, "HermiteBasis()")
    return nothing
end

support(basis::HermiteBasis) = RealInterval(-Inf, Inf)
