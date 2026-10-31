"""
    TransportMapDistribution(map::M) where {M <: AbstractTransportMap}

Represent a fitted `PolynomialMap`, `ComposedMap`, `LinearMap`, or `LaplaceMap`
as a continuous multivariate distribution. Standalone linear and Laplace maps
use a product standard-normal reference and map target to reference, so their
inverse generates the corresponding Gaussian samples. Compositions inherit the
polynomial layer's reference and the direction of the whole map.

`rand(d, n)` returns samples in columns, following Distributions.jl. Map-level
batch operations use rows. Statistics for standalone linear and Laplace maps
use exact Gaussian formulas by default. Other maps use reference-space
quadrature. An explicit `quad` keyword requests quadrature for any map.
Maps are retained by reference, so later parameter changes affect the
distribution. Use `deepcopy(map)` to freeze a fitted map.

### Example
```julia
map = PolynomialMap(2, 2) # initialize and fit map, optimize!() etc..
d = TransportMapDistribution(map)
# Sampling
samples = rand(d, 10)
# Moments
μ = mean(d)
σ = std(d)
Σ = cov(d)
# Density
dens = pdf(d, [1.0, -1.0])
```
"""
struct TransportMapDistribution{M <: AbstractTransportMap} <: Distributions.ContinuousMultivariateDistribution
    map::M

    function TransportMapDistribution(map::M) where {M <: AbstractTransportMap}
        numberdimensions(map) > 0 || throw(ArgumentError("A distribution requires a dimensioned map; use LinearMap(n) for a standalone identity"))
        _validate_distribution_map(map)
        return new{M}(map)
    end
end

function _validate_distribution_map(map::AbstractTransportMap)
    throw(ArgumentError("Distribution operations are not implemented for $(typeof(map))"))
end

function _validate_distribution_map(map::PolynomialMap)
    _reference_marginal(map) isa Distributions.ContinuousUnivariateDistribution ||
        throw(ArgumentError("TransportMapDistribution requires a continuous reference"))
    _map_direction(map) in (:reference, :target) || throw(ArgumentError("Invalid map direction"))
    return nothing
end

function _validate_distribution_map(map::LinearMap)
    length(map.μ) == length(map.σ) || throw(DimensionMismatch("Linear location and scale dimensions must agree"))
    all(isfinite, map.μ) && all(value -> isfinite(value) && !iszero(value), map.σ) ||
        throw(ArgumentError("Linear maps require finite locations and finite nonzero scales"))
    return nothing
end

function _validate_distribution_map(map::LaplaceMap)
    d = numberdimensions(map)
    size(map.chol) == (d, d) || throw(DimensionMismatch("Laplace location and factor dimensions must agree"))
    all(isfinite, map.mode) && all(isfinite, map.chol) && istril(map.chol) &&
        all(value -> !iszero(value), diag(map.chol)) ||
        throw(ArgumentError("Laplace maps require a finite location and nonsingular lower-triangular factor"))
    return nothing
end

function _validate_distribution_map(map::ComposedMap)
    _validate_distribution_map(map.linearmap)
    _validate_distribution_map(map.polynomialmap)
    numberdimensions(map.linearmap) in (0, numberdimensions(map.polynomialmap)) ||
        throw(DimensionMismatch("Composed map dimensions must agree"))
    return nothing
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
function _reference_distribution(tm::AbstractTransportMap)
    return product_distribution(fill(_reference_marginal(tm), numberdimensions(tm)))
end

function _from_reference(tm::AbstractTransportMap, x::AbstractVecOrMat{<:Real})
    if _map_direction(tm) == :target
        return evaluate(tm, x) # map from density
    else
        return inverse(tm, x) # map from samples
    end
end

function _to_reference(tm::AbstractTransportMap, x::AbstractVecOrMat{<:Real})
    if _map_direction(tm) == :target
        return inverse(tm, x) # map from density
    else
        return evaluate(tm, x) # map from samples
    end
end

function _nth_moment(tm::AbstractTransportMap, n::Int64, quad::AbstractQuadratureWeights)
    return (_from_reference(tm, quad.points) .^ n)' * quad.weights
end

function _centered_map_values(tm::AbstractTransportMap, quad::AbstractQuadratureWeights)
    X = _from_reference(tm, quad.points)
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
_statistics_quadrature(map::AbstractTransportMap) = SparseSmolyakWeights(3, map)
_statistics_quadrature(::Union{LinearMap, LaplaceMap}) = nothing

mean(d::TransportMapDistribution; quad::Union{Nothing, AbstractQuadratureWeights} = _statistics_quadrature(d.map)) =
    _map_mean(d.map, quad)
var(d::TransportMapDistribution; quad::Union{Nothing, AbstractQuadratureWeights} = _statistics_quadrature(d.map)) =
    _map_var(d.map, quad)
std(d::TransportMapDistribution; quad::Union{Nothing, AbstractQuadratureWeights} = _statistics_quadrature(d.map)) =
    _map_std(d.map, quad)
cov(d::TransportMapDistribution; quad::Union{Nothing, AbstractQuadratureWeights} = _statistics_quadrature(d.map)) =
    _map_cov(d.map, quad)
cor(d::TransportMapDistribution; quad::Union{Nothing, AbstractQuadratureWeights} = _statistics_quadrature(d.map)) =
    _map_cor(d.map, quad)
entropy(d::TransportMapDistribution; quad::Union{Nothing, AbstractQuadratureWeights} = _statistics_quadrature(d.map)) =
    _map_entropy(d.map, quad)

_map_mean(map::AbstractTransportMap, quad::AbstractQuadratureWeights) = _nth_moment(map, 1, quad)

function _map_var(map::AbstractTransportMap, quad::AbstractQuadratureWeights)
    centered = _centered_map_values(map, quad)
    return (centered .^ 2)' * quad.weights
end

_map_std(map::AbstractTransportMap, quad::AbstractQuadratureWeights) = sqrt.(_map_var(map, quad))

function _map_cov(map::AbstractTransportMap, quad::AbstractQuadratureWeights)
    centered = _centered_map_values(map, quad)
    return centered' * (quad.weights .* centered)
end

function _map_cor(map::AbstractTransportMap, quad::AbstractQuadratureWeights)
    C = _map_cov(map, quad)
    σ = sqrt.(diag(C))
    return C ./ (σ .* σ')
end

function _map_entropy(tm::AbstractTransportMap, quad::AbstractQuadratureWeights)
    points = _map_direction(tm) == :target ? quad.points :
        _from_reference(tm, quad.points)

    logdet = [logabsdetjacobian(tm, point) for point in eachrow(points)]
    correction = dot(quad.weights, logdet)
    H_reference = entropy(_reference_distribution(tm))

    return _map_direction(tm) == :target ?
        H_reference + correction :
        H_reference - correction
end

# These maps invert a standard normal reference into a Gaussian. Keep all six
# statistics independent of quadrature, and return fresh arrays for locations.
_map_mean(map::LinearMap, ::Nothing) = copy(map.μ)
_map_mean(map::LaplaceMap, ::Nothing) = copy(map.mode)
_map_var(map::LinearMap, ::Nothing) = map.σ .^ 2
_map_var(map::LaplaceMap, ::Nothing) = vec(sum(abs2, map.chol; dims = 2))
_map_std(map::LinearMap, ::Nothing) = abs.(map.σ)
_map_std(map::LaplaceMap, ::Nothing) = norm.(eachrow(map.chol))
_map_cov(map::LinearMap, ::Nothing) = Matrix(Diagonal(map.σ .^ 2))
_map_cov(map::LaplaceMap, ::Nothing) = cov(map)
_map_cor(map::LinearMap, ::Nothing) = Matrix{Float64}(I, numberdimensions(map), numberdimensions(map))

function _map_cor(map::LaplaceMap, ::Nothing)
    # Normalize the factor first to avoid squaring extreme scales in covariance.
    normalized = map.chol ./ _map_std(map, nothing)
    return normalized * normalized'
end

function _map_entropy(map::Union{LinearMap, LaplaceMap}, ::Nothing)
    scales = map isa LinearMap ? map.σ : diag(map.chol)
    return numberdimensions(map) * entropy(Normal()) + sum(value -> log(abs(value)), scales)
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
