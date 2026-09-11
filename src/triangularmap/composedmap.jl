"""
    ComposedMap{T <: AbstractLinearMap} <: AbstractComposedMap

A composed transport map consisting of a linear map followed by a polynomial map.

The composition is ``S(x)=M(L(x))``, where `L` is a linear map of type `T` and
`M` is the polynomial map. The linear map can be a `LinearMap` or `LaplaceMap`.

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

"""
    pullback(C::ComposedMap, x::AbstractVector{<:Real})

Compute the pullback density of the composed map:

```math
\\operatorname{pullback}(C,x)
= \\operatorname{pullback}(M,L(x))\\,|\\det\\nabla L(x)|
= \\frac{\\operatorname{pullback}(M,L(x))}{\\prod_i\\sigma_i}.
```
"""
function pullback(C::ComposedMap{T}, x::AbstractVector{<:Real}) where {T <: AbstractLinearMap}
    y = evaluate(C.linearmap, x)
    return pullback(C.polynomialmap, y) ./ jacobian(C.linearmap)  # Adjust for map scaling
end

"""
    pullback(C::ComposedMap, X::AbstractMatrix{<:Real})

Compute the pullback density for multiple points (row-wise).
"""
function pullback(C::ComposedMap{T}, X::AbstractMatrix{<:Real}) where {T <: AbstractLinearMap}
    Y = evaluate(C.linearmap, X)
    return pullback(C.polynomialmap, Y) ./ jacobian(C.linearmap)  # Adjust for map scaling
end

"""
    numberdimensions(C::ComposedMap)

Return the number of dimensions of the composed map.
"""
numberdimensions(C::ComposedMap{T}) where {T <: AbstractLinearMap} = numberdimensions(C.linearmap)

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
