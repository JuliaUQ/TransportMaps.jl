"""
    InverseMap(map::AbstractTransportMap)
    inv(map::AbstractTransportMap)
    map^-1

Wrap a transport map so that its inverse is evaluated as the forward map.
For `I = inv(map)`, `evaluate(I, x)` calls `inverse(map, x)`, and
`inverse(I, x)` calls `evaluate(map, x)`. Vectors represent single points;
matrix inputs contain points in rows.

The wrapper shares the original map rather than copying it, and `inv(I)`
returns that map. Inversion inherits its domain restrictions and numerical
accuracy. Point-dependent Jacobian methods use the inverse transformation's
determinant.

An `InverseMap` can be used as a layer in a [`ComposedMap`](@ref), but cannot
currently be wrapped directly in a `TransportMapDistribution`.

# Example
```julia
L = LinearMap([1.0], [2.0])
I = L^-1
evaluate(I, [0.5])  # [2.0]
inv(I) === L         # true
```
"""
struct InverseMap{T <: AbstractTransportMap} <: AbstractTransportMap
    map::T
end

Base.inv(map::AbstractTransportMap) = InverseMap(map)
Base.inv(map::InverseMap) = map.map
Base.literal_pow(::typeof(^), map::AbstractTransportMap, ::Val{-1}) = inv(map)

_has_forwarddirection(map::AbstractTransportMap) =
    applicable(forwarddirection, map) || hasproperty(map, :forwarddirection) ||
    applicable(_map_direction, map)
_has_forwarddirection(map::InverseMap) = _has_forwarddirection(map.map)

function forwarddirection(map::InverseMap)
    wrapped = map.map
    if applicable(forwarddirection, wrapped)
        direction = forwarddirection(wrapped)
    elseif hasproperty(wrapped, :forwarddirection)
        direction = wrapped.forwarddirection
    elseif applicable(_map_direction, wrapped)
        direction = _map_direction(wrapped)
    else
        throw(ArgumentError("Direction is not defined for $(typeof(wrapped))"))
    end
    direction in (:reference, :target) || throw(ArgumentError("Invalid map direction"))
    return direction === :reference ? :target : :reference
end

numberdimensions(map::InverseMap) = numberdimensions(map.map)

evaluate(map::InverseMap, x::AbstractVecOrMat{<:Real}) =
    inverse(map.map, x)

inverse(map::InverseMap, x::AbstractVecOrMat{<:Real}) =
    evaluate(map.map, x)

function jacobian(map::InverseMap, x::AbstractVector{<:Real})
    return inv(jacobian(map.map, inverse(map.map, x)))
end

function jacobian(map::InverseMap, X::AbstractMatrix{<:Real})
    return inv.(jacobian(map.map, inverse(map.map, X)))
end

logabsdetjacobian(map::InverseMap, x::AbstractVector{<:Real}) =
    -logabsdetjacobian(map.map, inverse(map.map, x))

logabsdetjacobian(map::InverseMap, X::AbstractMatrix{<:Real}) =
    -logabsdetjacobian(map.map, inverse(map.map, X))

logpullback(map::InverseMap, density::MapTargetDensity, x::AbstractVector{<:Real}) =
    logpushforward(map.map, density, x)

logpullback(map::InverseMap, density::MapTargetDensity, X::AbstractMatrix{<:Real}) =
    logpushforward(map.map, density, X)

logpushforward(map::InverseMap, x::AbstractVector{<:Real}) =
    logpullback(map.map, x)

logpushforward(map::InverseMap, X::AbstractMatrix{<:Real}) =
    logpullback(map.map, X)

pushforward(map::InverseMap, x::AbstractVecOrMat{<:Real}) =
    pullback(map.map, x)

pullback(map::InverseMap, density::MapTargetDensity, x::AbstractVecOrMat{<:Real}) =
    pushforward(map.map, density, x)

(map::InverseMap)(x::AbstractVecOrMat{<:Real}) = evaluate(map, x)

# _reference_distribution(map::InverseMap) = _reference_distribution(map.map)
# _reference_marginal(map::InverseMap) = _reference_marginal(map.map)
# _map_direction(map::InverseMap) = _map_direction(map.map)

# _inverse_coordinate_functions(map::InverseMap, previous, target, k) = _inverse_coordinate_functions(map.map, previous, target, k)
