"""
    MultivariateBasis{T <: AbstractPolynomialBasis}

A multivariate polynomial basis function constructed as a tensor product of univariate polynomial bases.

# Fields
- `multiindexset::Vector{Int}`: Multi-index representing the polynomial degrees for each dimension
- `univariatebases::Vector{T}`: Vector of univariate basis functions, one for each dimension

# Constructors
- `MultivariateBasis(multiindexset::Vector{Int}, ::Type{T}) where T<:AbstractPolynomialBasis`: Create a multivariate basis with the specified multi-index and basis type.
- `MultivariateBasis(multiindexset::Vector{Int}, basistype::AbstractPolynomialBasis)`: Constructor that accepts a basis instance instead of a type.
"""
struct MultivariateBasis{T <: AbstractPolynomialBasis} <: AbstractMultivariateBasis
    multiindexset::Vector{Int}
    univariatebases::Vector{T}  # Store univariate basis for each dimension
end

function MultivariateBasis(multiindexset::Vector{Int}, ::Type{T}) where {T <: AbstractPolynomialBasis}
    if T <: LinearizedHermiteBasis
        univariatebases = [T(degree) for degree in multiindexset]
    else
        univariatebases = [T() for _ in multiindexset]
    end
    return MultivariateBasis{T}(multiindexset, univariatebases)
end

# Backwards-compatible constructor that accepts a basis instance
MultivariateBasis(multiindexset::Vector{Int}, basistype::AbstractPolynomialBasis) = MultivariateBasis(multiindexset, typeof(basistype))

"""
    Psi(alpha::Vector{<:Real}, z::Vector{<:Real}, univariatebases::Vector{T}) where {T <: AbstractPolynomialBasis}

Evaluate the tensor product of univariate basis functions.

Returns ``\\prod_i \\psi_i(\\alpha_i, z_i)``.
"""
function Psi(alpha::Vector{<:Real}, z::Vector{<:Real}, univariatebases::Vector{T}) where {T <: AbstractPolynomialBasis}
    @assert length(alpha) == length(z) "Dimension mismatch: alpha and z must have same length"
    return prod(basisfunction(ub, αᵢ, zᵢ) for (αᵢ, zᵢ, ub) in zip(alpha, z, univariatebases))
end

"""
    evaluate(mvb::MultivariateBasis{T}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}

Evaluate the multivariate basis function at point z.
"""
function evaluate(mvb::MultivariateBasis{T}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}
    @assert length(mvb.multiindexset) == length(z) "Dimension mismatch: multiindexset and z must have same length"
    alpha = Real.(mvb.multiindexset)
    return Psi(alpha, z, mvb.univariatebases)
end

"""
    f(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}

Evaluate the linear combination
``f(z) = \\sum_i c_i \\Psi_i(z)``.
"""
function f(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}
    @assert length(Ψ) == length(coefficients) "Number of basis functions must equal number of coefficients"
    return sum(coeff * evaluate(mvb, z) for (coeff, mvb) in zip(coefficients, Ψ))
end

"""
    f(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}) where {T <: AbstractPolynomialBasis}

Return a closure representing ``z \\mapsto f(\\Psi, c, z)``.
"""
function f(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}) where {T <: AbstractPolynomialBasis}
    return (z::Vector{<:Real}) -> f(Ψ, coefficients, z)
end

"""
    gradient_z(mvb::MultivariateBasis{T}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}

Compute the gradient ``\\nabla_z \\Psi(z)`` of the multivariate basis.
"""
function gradient_z(mvb::MultivariateBasis{T}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}
    return [partial_derivative_z(mvb.univariatebases, mvb.multiindexset, z, j) for j in 1:length(z)]
end

"""
    partial_derivative_z(bases::Vector{T}, α::Vector{Int}, z::Vector{<:Real}, j::Int) where {T <: AbstractPolynomialBasis}

Compute the partial derivative ``\\partial \\Psi / \\partial z_j`` of the tensor-product basis.
"""
function partial_derivative_z(bases::Vector{T}, α::Vector{Int}, z::Vector{<:Real}, j::Int) where {T <: AbstractPolynomialBasis}
    @assert 1 <= j <= length(z) "Index j must be within bounds of z"
    @assert length(bases) == length(z) "Dimension mismatch"

    # Compute the product of all terms except the j-th, times the derivative of the j-th term
    result = basisfunction_derivative(bases[j], α[j], z[j])
    for (i, (αᵢ, zᵢ)) in enumerate(zip(α, z))
        if i != j
            result *= basisfunction(bases[i], αᵢ, zᵢ)
        end
    end
    return result
end

"""
    partial_derivative_z(mvb::MultivariateBasis{T}, z::Vector{<:Real}, j::Int) where {T <: AbstractPolynomialBasis}

Compute ``\\partial \\Psi / \\partial z_j`` for the multivariate basis.
"""
partial_derivative_z(mvb::MultivariateBasis{T}, z::Vector{<:Real}, j::Int) where {T <: AbstractPolynomialBasis} = partial_derivative_z(mvb.univariatebases, mvb.multiindexset, z, j)

"""
    partial_derivative_z(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}, z::Vector{<:Real}, j::Int) where {T <: AbstractPolynomialBasis}

Compute
``\\partial f / \\partial z_j = \\sum_i c_i\\,\\partial \\Psi_i / \\partial z_j``.
"""
function partial_derivative_z(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}, z::Vector{<:Real}, j::Int) where {T <: AbstractPolynomialBasis}
    @assert length(Ψ) == length(coefficients) "Number of basis functions must equal number of coefficients"
    return sum(coeff * partial_derivative_z(mvb, z, j) for (coeff, mvb) in zip(coefficients, Ψ))
end

"""
    gradient_z(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}

Compute the gradient ``\\nabla_z f(z)``.
"""
function gradient_z(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}
    return [partial_derivative_z(Ψ, coefficients, z, j) for j in 1:length(z)]
end

"""
    gradient_coefficients(Ψ::Vector{MultivariateBasis{T}}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}

Compute the coefficient gradient
``\\nabla_c f(z) = [\\Psi_1(z), \\Psi_2(z), \\ldots]^\\mathsf{T}``.
"""
function gradient_coefficients(Ψ::Vector{MultivariateBasis{T}}, z::Vector{<:Real}) where {T <: AbstractPolynomialBasis}
    return [evaluate(mvb, z) for mvb in Ψ]
end

"""
    basistype(mvb::MultivariateBasis{T}) where {T <: AbstractPolynomialBasis}

Return the type of the univariate basis used in the multivariate basis.
"""
function basistype(mvb::MultivariateBasis{T}) where {T <: AbstractPolynomialBasis}
    return T
end

# Display methods for MultivariateBasis
function Base.show(io::IO, basis::MultivariateBasis{T}) where {T <: AbstractPolynomialBasis}
    basis_name = string(basistype(basis))


    degree = sum(basis.multiindexset)
    dimension = length(basis.multiindexset)

    print(io, "MultivariateBasis(")
    print(io, "$(basis.multiindexset), ")
    print(io, "degree=$degree, ")
    print(io, "dim=$dimension, ")
    print(io, "basis=$basis_name)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", basis::MultivariateBasis{T}) where {T <: AbstractPolynomialBasis}
    basis_name = string(basistype(basis))

    degree = sum(basis.multiindexset)
    dimension = length(basis.multiindexset)

    println(io, "MultivariateBasis:")
    println(io, "  Multi-index: $(basis.multiindexset)")
    println(io, "  Total degree: $degree")
    println(io, "  Dimension: $dimension")
    println(io, "  Basis type: $basis_name")

    # Show individual polynomial degrees for each dimension
    if dimension > 1
        println(io, "  Individual degrees: $(basis.multiindexset)")
    end
    return nothing
end
