# Distributions interface for fitted transport map
struct TransportMapDistribution{M <: AbstractTransportMap} <: Distributions.ContinuousMultivariateDistribution
    map::M
end

function Base.show(io::IO, d::TransportMapDistribution)
    print(io, "TransportMapDistribution(")
    print(io, "map=$(d.map)")
    return print(io, ")")
end

Base.length(d::TransportMapDistribution) = numberdimensions(d.map)
Distributions.sampler(d::TransportMapDistribution) = d
Base.eltype(::Type{<:TransportMapDistribution}) = Float64

## Helper functions
function _reference_distribution(pm::PolynomialMap)
    return product_distribution(fill(pm.reference.densitytype, numberdimensions(pm)))
end

function _from_reference(pm::PolynomialMap, x::AbstractVecOrMat{<:Real})
    if pm.forwarddirection == :target
        return evaluate(pm, x) # map from density
    else
        return inverse(pm, x) # map from samples
    end
end

function _to_reference(pm::PolynomialMap, x::AbstractVecOrMat{<:Real})
    if pm.forwarddirection == :target
        return inverse(pm, x) # map from density
    else
        return evaluate(pm, x) # map from samples
    end
end

function _nth_moment(pm::PolynomialMap, n::Int64, quad::AbstractQuadratureWeights)
    return (_from_reference(pm, quad.points) .^ n)' * quad.weights
end

function _centered_map_values(pm::PolynomialMap, quad::AbstractQuadratureWeights)
    X = _from_reference(pm, quad.points)
    μ = X' * quad.weights
    return X .- μ'
end

## Sampling
function Distributions._rand!(
        rng::AbstractRNG,
        d::TransportMapDistribution,
        x::AbstractVector{<:Real},
    )
    z = rand(rng, _reference_distribution(d.map))
    copyto!(x, _from_reference(d.map, z))
    return x
end

function Distributions._rand!(
        rng::AbstractRNG,
        d::TransportMapDistribution,
        X::AbstractMatrix{<:Real}
    )
    Z = rand(rng, _reference_distribution(d.map), size(X, 2))
    copyto!(X, permutedims(_from_reference(d.map, permutedims(Z))))
    return X
end

## (log)pdf
function Distributions._logpdf(d::TransportMapDistribution, x::AbstractVector{<:Real})
    return logpullback(d.map, x)
end

## Moments
function mean(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    return _nth_moment(d.map, 1, quad)
end

function var(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    centered = _centered_map_values(d.map, quad)
    return (centered .^ 2)' * quad.weights
end

function std(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    return sqrt.(var(d; quad = quad))
end

function cov(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    centered = _centered_map_values(d.map, quad)
    return centered' * (quad.weights .* centered)
end

function cor(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    C = cov(d; quad = quad)
    σ = sqrt.(diag(C))
    return C ./ (σ .* σ')
end

function entropy(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    pm = d.map
    points = pm.forwarddirection == :target ? quad.points :
        _from_reference(pm, quad.points)

    logdet = [_logabsdetjacobian(pm, point) for point in eachrow(points)]
    correction = dot(quad.weights, logdet)
    H_reference = entropy(_reference_distribution(pm))

    return pm.forwarddirection == :target ?
        H_reference + correction :
        H_reference - correction
end

## Support
"""
    insupport(d::TransportMapDistribution, x::AbstractVector{<:Real})

Check whether `x` belongs to the support induced by the transport map. The point
must have the distribution's dimension and finite coordinates.

Maps from samples are evaluated in reference coordinates. Maps from density are
inverted componentwise, checking the reference interval before each solve. This
also handles supports that are not rectangular. Comparisons at finite endpoints
allow floating-point roundoff. Global coordinate bounds are not inferred from
the reference distribution's bounds.
"""
function insupport(d::TransportMapDistribution, x::AbstractVector{<:Real})
    return !isnothing(_reference_point_in_support(d.map, x))
end
