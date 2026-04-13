"""
    LinearizedHermiteBasis

Probabilist Hermite polynomial basis with linearization outside specified bounds.

# Fields
- `linearizationbounds::Vector{Float64}`: lower and upper bounds for linearization.
- `normalization::Vector{Float64}`: normalization factors for each degree.

# Constructors
- `LinearizedHermiteBasis(; lower=-Inf, upper=Inf, normalization=Float64[])`: Explicit constructor with specified bounds and normalization.
- `LinearizedHermiteBasis(max_degree::Int)`: Construct with default normalization for given maximum degree.
- `LinearizedHermiteBasis(samples::Vector{<:Real}, max_degree::Int, k::Int)`: Construct bounds from 1st and 99th percentile of samples.
- `LinearizedHermiteBasis(density::Distributions.UnivariateDistribution, max_degree::Int, k::Int)`: Construct bounds from 1st and 99th percentile of reference density.
"""
struct LinearizedHermiteBasis <: AbstractPolynomialBasis
    linearizationbounds::Vector{Float64}
    normalization::Vector{Float64}

    function LinearizedHermiteBasis(; lower=-Inf, upper=Inf, normalization=Float64[])
        return new([lower, upper], normalization)
    end
end

LinearizedHermiteBasis(max_degree::Int) = LinearizedHermiteBasis(normalization=ones(max_degree + 1))

function LinearizedHermiteBasis(samples::Vector{<:Real}, max_degree::Int, k::Int)
    lower_bound, upper_bound = quantile(samples, 0.01), quantile(samples, 0.99)
    normalization = [factorial(n) for n in 0:max_degree]
    if k <= max_degree
        normalization[k+1] = factorial(k + 1)
    end
    return LinearizedHermiteBasis(lower=lower_bound, upper=upper_bound, normalization=normalization)
end

function LinearizedHermiteBasis(density::Distributions.UnivariateDistribution, max_degree::Int, k::Int)
    lower_bound, upper_bound = quantile(density, 0.01), quantile(density, 0.99)
    normalization = [factorial(n) for n in 0:max_degree]
    if k <= max_degree
        normalization[k+1] = factorial(k + 1)
    end
    return LinearizedHermiteBasis(lower=lower_bound, upper=upper_bound, normalization=normalization)
end

@inline function _linearized_hermite(n::Int, z::Real, linearizationbounds::Vector{Float64})
    lower = linearizationbounds[1]
    upper = linearizationbounds[2]
    if z < lower
        φ = hermite_polynomial(n, lower)
        dφ = hermite_derivative(n, lower)
        return φ + dφ * (z - lower)
    elseif z > upper
        φ = hermite_polynomial(n, upper)
        dφ = hermite_derivative(n, upper)
        return φ + dφ * (z - upper)
    else
        return hermite_polynomial(n, z)
    end
end

@inline function _linearized_hermite_derivative(n::Int, z::Real, linearizationbounds::Vector{Float64})
    lower = linearizationbounds[1]
    upper = linearizationbounds[2]
    if z < lower
        return hermite_derivative(n, lower)
    elseif z > upper
        return hermite_derivative(n, upper)
    else
        return hermite_derivative(n, z)
    end
end

"""
    basisfunction(basis::LinearizedHermiteBasis, αᵢ::Int, zᵢ::Real)

Evaluate `LinearizedHermiteBasis` with degree `αᵢ` at `zᵢ`.
"""
@inline function basisfunction(basis::LinearizedHermiteBasis, αᵢ::Int, zᵢ::Real)
    n = Int(αᵢ)
    if !isempty(basis.normalization) && isfinite(basis.linearizationbounds[1]) && isfinite(basis.linearizationbounds[2])
        return _linearized_hermite(n, zᵢ, basis.linearizationbounds) / sqrt(basis.normalization[n+1])
    else
        return _linearized_hermite(n, zᵢ, basis.linearizationbounds)
    end
end

"""
    basisfunction_derivative(basis::LinearizedHermiteBasis, αᵢ::Int, zᵢ::Real)

Evaluate derivative of `LinearizedHermiteBasis` with degree `αᵢ` at `zᵢ`.
"""
@inline function basisfunction_derivative(basis::LinearizedHermiteBasis, αᵢ::Int, zᵢ::Real)
    n = Int(αᵢ)
    if !isempty(basis.normalization) && isfinite(basis.linearizationbounds[1]) && isfinite(basis.linearizationbounds[2])
        return _linearized_hermite_derivative(n, zᵢ, basis.linearizationbounds) / sqrt(basis.normalization[n+1])
    else
        return _linearized_hermite_derivative(n, zᵢ, basis.linearizationbounds)
    end
end

function Base.show(io::IO, basis::LinearizedHermiteBasis)
    print(io, "LinearizedHermiteBasis(bounds=$(basis.linearizationbounds))")
end

support(basis::LinearizedHermiteBasis) = RealInterval(-Inf, Inf)
