struct InverseMap{T <: AbstractTransportMap} <: AbstractTransportMap
    map::T
end

Base.inv(map::AbstractTransportMap) = InverseMap(map)
Base.inv(map::InverseMap) = map.map
Base.literal_pow(::typeof(^), map::AbstractTransportMap, ::Val{-1}) = inv(map)

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

pushforward!(map::InverseMap, x::AbstractVecOrMat{<:Real}) =
    pullback!(map.map, x)

pullback(map::InverseMap, density::MapTargetDensity, x::AbstractVecOrMat{<:Real}) =
    pushforward(map.map, density, x)

pullback!(map::InverseMap, density::MapTargetDensity, x::AbstractVecOrMat{<:Real}) =
    pushforward!(map.map, density, x)

(map::InverseMap)(x::AbstractVecOrMat{<:Real}) = evaluate(map, x)

# _reference_distribution(map::InverseMap) = _reference_distribution(map.map)
# _reference_marginal(map::InverseMap) = _reference_marginal(map.map)
# _map_direction(map::InverseMap) = _map_direction(map.map)

# _inverse_coordinate_functions(map::InverseMap, previous, target, k) = _inverse_coordinate_functions(map.map, previous, target, k)
