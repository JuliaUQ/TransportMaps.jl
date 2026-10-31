"""
    ComposedMap{T <: AbstractLinearMap} <: AbstractComposedMap

A composed transport map consisting of a linear map followed by a polynomial map.

The composition is ``S(x)=M(L(x))``, where `L` is a linear map of type `T` and
`M` is the polynomial map. The linear map can be a `LinearMap` or `LaplaceMap`. The direction of the whole
composition is inherited from the polynomial map: `:reference` means target to
reference and `:target` means reference to target. Its reference distribution is
the endpoint reference of the whole map, before both layers for `:target` and
after both layers for `:reference`.

# Fields
- `linearmap<:T`: The linear map component
- `polynomialmap::PolynomialMap`: The polynomial map component

# Constructors
- `ComposedMap(lm::AbstractLinearMap, pm::PolynomialMap)`
"""
struct ComposedMap{T <: AbstractLinearMap} <: AbstractComposedMap
    linearmap::T
    polynomialmap::PolynomialMap

    function ComposedMap(lm::AbstractLinearMap, pm::PolynomialMap)
        if numberdimensions(lm) == 0
            T = typeof(lm)
            return new{T}(lm, pm)
        else
            @assert numberdimensions(lm) == numberdimensions(pm) "Linear map and polynomial map must have the same number of dimensions"
            T = typeof(lm)
            return new{T}(lm, pm)
        end
    end
end

"""
    evaluate(C::ComposedMap, x::AbstractVector{<:Real})

Evaluate the composed map, ``S(x)=M(L(x))``.
"""
function evaluate(C::ComposedMap{T}, x::AbstractVector{<:Real}) where {T <: AbstractLinearMap}
    y = evaluate(C.linearmap, x)
    return evaluate(C.polynomialmap, y)
end

"""
    evaluate(C::ComposedMap, X::AbstractMatrix{<:Real})

Evaluate the composed map for multiple points (row-wise).
"""
function evaluate(C::ComposedMap{T}, X::AbstractMatrix{<:Real}) where {T <: AbstractLinearMap}
    Y = evaluate(C.linearmap, X)
    return evaluate(C.polynomialmap, Y)
end

"""
    inverse(C::ComposedMap, z::AbstractVector{<:Real})

Invert the composed map, ``S^{-1}(z)=L^{-1}(M^{-1}(z))``.
"""
function inverse(C::ComposedMap{T}, z::AbstractVector{<:Real}) where {T <: AbstractLinearMap}
    y = inverse(C.polynomialmap, z)
    return inverse(C.linearmap, y)
end

"""
    inverse(C::ComposedMap, Z::AbstractMatrix{<:Real})

Invert the composed map for multiple points (row-wise).
"""
function inverse(C::ComposedMap{T}, Z::AbstractMatrix{<:Real}) where {T <: AbstractLinearMap}
    Y = inverse(C.polynomialmap, Z)
    return inverse(C.linearmap, Y)
end

# Keep composition metadata behind accessors so distribution operations do not
# depend on this two-layer container's fields. The whole map is M ∘ L, matching
# the evaluation-order and direction conventions of the map-chains branch.
_reference_marginal(C::ComposedMap) = _reference_marginal(C.polynomialmap)
_map_direction(C::ComposedMap) = _map_direction(C.polynomialmap)

function logabsdetjacobian(C::ComposedMap, x::AbstractVector{<:Real})
    return logabsdetjacobian(C.linearmap, x) +
        logabsdetjacobian(C.polynomialmap, evaluate(C.linearmap, x))
end

function jacobian(C::ComposedMap, x::AbstractVector{<:Real})
    return jacobian(C.linearmap, x) * jacobian(C.polynomialmap, evaluate(C.linearmap, x))
end

function jacobian(C::ComposedMap, X::AbstractMatrix{<:Real})
    @assert size(X, 2) == numberdimensions(C) "Input dimension must match the map"
    return [jacobian(C, x) for x in eachrow(X)]
end

function _inverse_coordinate_functions(C::ComposedMap, previous, target, k)
    transformed, slope = _linear_prefix(C.linearmap, previous, k)
    fun, derivative = _inverse_coordinate_functions(
        C.polynomialmap, view(transformed, 1:(k - 1)), target, k,
    )
    # The input to polynomial component k is affine in the original coordinate.
    # Orient the residual upwards even for a signed linear scale, so guarded
    # bracketing always uses the reference interval of the whole composition.
    orientation = sign(slope)
    coordinate(t) = muladd(slope, t, transformed[k])
    residual(t) = orientation * fun(coordinate(t))
    gradient(t) = abs(slope) * derivative(coordinate(t))
    return residual, gradient
end

"""
    numberdimensions(C::ComposedMap)

Return the number of dimensions of the composed map.
"""
numberdimensions(C::ComposedMap{T}) where {T <: AbstractLinearMap} = numberdimensions(C.polynomialmap)

function Base.show(io::IO, C::ComposedMap{T}) where {T <: AbstractLinearMap}
    print(io, "ComposedMap{$(T)}(")
    print(io, "linearmap=$(C.linearmap), ")
    print(io, "polynomialmap=$(C.polynomialmap)")
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", C::ComposedMap{T}) where {T <: AbstractLinearMap}
    println(io, "ComposedMap{$(T)} with $(numberdimensions(C)) dimensions:")
    println(io, " linearmap: ", C.linearmap)
    println(io, " polynomialmap: ", C.polynomialmap)
    return nothing
end
