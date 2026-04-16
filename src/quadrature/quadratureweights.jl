# Implementation of various quadrature rules for numerical integration

"""
    TensorProductWeights{T<:AbstractQuadratureKnots}

Multi-dimensional quadrature rule constructed from a tensor-product of one-dimensional
quadrature knots. The `level` parameter controls accuracy.

# Fields
- `points::Matrix{Float64}`: Quadrature points (tensor grid)
- `weights::Vector{Float64}`: Quadrature weights
- `knots::AbstractQuadratureKnots`: The 1D quadrature knots, e.g., `GaussHermiteKnots` or `GaussLegendreKnots`

# Constructors
- `TensorProductWeights(level::Int64, dim::Int64, knots::AbstractQuadratureKnots)`
- `TensorProductWeights(level::Int64, map::AbstractTransportMap)`

See also [`GaussHermiteWeights`](@ref) and [`GaussLegendreWeights`](@ref), and [`SparseSmolyakWeights`](@ref) for spare grids.
"""
struct TensorProductWeights{T<:AbstractQuadratureKnots} <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}
    knots::AbstractQuadratureKnots

    # Generic full tensor product basis
    function TensorProductWeights(level::Int64, dim::Int64, knots::AbstractQuadratureKnots)
        T = typeof(knots)
        points, weights = full_tensor_points(dim, level, knots)
        return new{T}(points, weights, knots)
    end

    function TensorProductWeights(level::Int64, map::AbstractTransportMap)
        dim = numberdimensions(map)
        knots = _determine_knots_from_reference(map)

        points, weights = full_tensor_points(dim, level, knots)
        return new{T}(points, weights, knots)
    end
end

Base.eltype(::Type{TensorProductWeights{T}}) where T<:AbstractQuadratureKnots = T

"""
    GaussHermiteWeights(level::Int64, dim::Int64)

Tensor-product Gauss-Hermite weights for integration with respect to standard Gaussian.
Returns a [`TensorProductWeights`](@ref) object.
"""
function GaussHermiteWeights(level::Int64, dim::Int64)
    # convenience constructor and compatibility
    return TensorProductWeights(level, dim, GaussHermiteKnots())
end

"""
    GaussHermiteWeights(level::Int64, map::AbstractTransportMap)

Tensor-product Gauss-Hermite weights for integration with respect to standard Gaussian
constructed from the transport map `map` with normal reference density.
Returns a [`TensorProductWeights`](@ref) object.
"""
function GaussHermiteWeights(level::Int64, map::AbstractTransportMap)
    if !(map.reference.densitytype isa Normal)
        error("GaussHermiteWeights requires Normal reference distribution, got $(typeof(ref_dist))")
    end

    return TensorProductWeights(level, numberdimensions(map), GaussHermiteKnots())
end

"""
    GaussLegendreWeights(level::Int64, dim::Int64, domain=[0, 1])

Tensor-product Gauss-Legendre weights for integration with respect to Uniform[0,1].
Returns a [`TensorProductWeights`](@ref) object.
"""
function GaussLegendreWeights(level::Int64, dim::Int64, domain::Vector{<:Real}=[0, 1])
    return TensorProductWeights(level, dim, GaussLegendreKnots(domain))
end

"""
    GaussLegendreWeights(level::Int64, map::AbstractTransportMap)

Tensor-product Gauss-Legendre weights for integration with respect to Uniform[a,b]
constructed from the transport map `map` with uniform reference density.
Returns a [`TensorProductWeights`](@ref) object.
"""
function GaussLegendreWeights(level::Int64, map::AbstractTransportMap)

    if !(map.reference.densitytype isa Uniform)
        error("GaussLegendreWeights requires Uniform reference distribution, got $(typeof(ref_dist))")
    end

    return TensorProductWeights(level, numberdimensions(map), _determine_knots_from_reference(ref))
end

"""
    SparseSmolyakWeights{T<:AbstractQuadratureKnots}

Multi-dimensional quadrature rule constructed using a sparse Smolyak grid of one-dimensional
quadrature knots. The `level` parameter controls accuracy.

# Fields
- `points::Matrix{Float64}`: Quadrature points (sparse grid)
- `weights::Vector{Float64}`: Quadrature weights
- `knots::AbstractQuadratureKnots`: The 1D quadrature knots, e.g., `GaussHermiteKnots` or `GaussLegendreKnots`

# Constructors
- `SparseSmolyakWeights(level::Int64, dim::Int64, knots=GaussHermiteKnots())`: Construct sparse Smolyak grid with specified `level`, `dim` and `knots`.
- `SparseSmolyakWeights(level::Int64, map::AbstractTransportMap)`: Construct sparse Smolyak grid with the help of the transport map. The knots are chosen based on the reference distribution ([`GaussHermiteKnots`](@ref) for a Normal reference and [`GaussLegendreKnots`](@ref) for a Uniform reference).

See also [`TensorProductWeights`](@ref).
"""
struct SparseSmolyakWeights{T<:AbstractQuadratureKnots} <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}
    knots::AbstractQuadratureKnots

    function SparseSmolyakWeights(level::Int64, dim::Int64, knots::AbstractQuadratureKnots=GaussHermiteKnots(); sparse::Bool=true)
        T = typeof(knots)
        points, weights = smolyak_points(dim, level, knots, sparse)
        return new{T}(points, weights, knots)
    end

    function SparseSmolyakWeights(level::Int64, map::AbstractTransportMap; sparse::Bool=true)
        dim = numberdimensions(map)
        knots = _determine_knots_from_reference(map)
        T = typeof(knots)
        points, weights = smolyak_points(dim, level, knots, sparse)

        return new{T}(points, weights, knots)
    end
end

Base.eltype(::Type{SparseSmolyakWeights{T}}) where T<:AbstractQuadratureKnots = T

function _determine_knots_from_reference(map::AbstractTransportMap)
    ref = map.reference.densitytype
    if ref isa Normal
        return GaussHermiteKnots()
    elseif ref isa Uniform
        return GaussLegendreKnots(support(ref))
    end
end

"""
    MonteCarloWeights

Monte Carlo quadrature using random samples from the reference distribution.
All points receive uniform weights `1/numberpoints`.

# Fields
- `points::Matrix{Float64}`: Quadrature points (random samples)
- `weights::Vector{Float64}`: Quadrature weights (uniform)

# Constructors
- `MonteCarloWeights(numberpoints::Int64, dimension::Int64)`: Construct weights with random sampling from `Normal()`.
- `MonteCarloWeights(numberpoints::Int64, map::AbstractTransportMap)`: Get number of dimensions and sample from the map's reference density.
- `MonteCarloWeights(points::Matrix{Float64}, weights::Vector{Float64}=Float64[])`: Construct from custom points and weights.
"""
struct MonteCarloWeights <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}

    function MonteCarloWeights(numberpoints::Int64, dimension::Int64)
        points, weights = montecarlo_weights(numberpoints, dimension)
        return new(points, weights)
    end

    function MonteCarloWeights(numberpoints::Int64, map::AbstractTransportMap)
        # Generate random points in the reference space
        points, weights = montecarlo_weights(numberpoints, numberdimensions(map), map.reference.densitytype)

        return new(points, weights)
    end

    function MonteCarloWeights(points::Matrix{Float64}, weights::Vector{Float64}=Float64[])
        if isempty(weights)
            # If no weights are provided, assume uniform weights
            weights = 1 / size(points, 1) * ones(size(points, 1))
        end
        return new(points, weights)
    end
end

function montecarlo_weights(numberpoints::Int64, dimension::Int64, distr::Distributions.UnivariateDistribution=Normal())
    points = rand(distr, numberpoints, dimension)
    weights = 1 / numberpoints * ones(numberpoints)
    return points, weights
end

"""
    LatinHypercubeWeights

Latin Hypercube sampling for quasi-Monte Carlo integration. Provides better
space-filling properties than pure Monte Carlo. All points receive uniform
weights `1/n`.

# Fields
- `points::Matrix{Float64}`: Quadrature points (Latin Hypercube samples)
- `weights::Vector{Float64}`: Quadrature weights (uniform)

# Constructors
- `LatinHypercubeWeights(n::Int64, d::Int64)`: Construct Latin Hypercube samples for `d` dimensions using `Normal()`.
- `LatinHypercubeWeights(n::Int64, map::AbstractTransportMap)`: Get number of dimensions and sample according to the map's reference density.
"""
struct LatinHypercubeWeights <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}

    function LatinHypercubeWeights(n::Int64, d::Int64)
        points, weights = latinhypercube_weights(n, d)
        return new(points, weights)
    end

    function LatinHypercubeWeights(n::Int64, map::AbstractTransportMap)
        # Generate Latin Hypercube points in the reference space
        points, weights = latinhypercube_weights(n, numberdimensions(map), map.reference.densitytype)
        return new(points, weights)
    end
end

function latinhypercube_weights(numberpoints::Int64, dimension::Int64, distr::Distributions.UnivariateDistribution=Normal())
    points = reshape([quantile(distr, u) for u in QuasiMonteCarlo.sample(numberpoints, dimension, LatinHypercubeSample())], numberpoints, dimension)
    weights = 1 / numberpoints * ones(numberpoints)
    return points, weights
end

# Display methods for TensorProductWeights
function Base.show(io::IO, w::TensorProductWeights{T}) where T<:AbstractQuadratureKnots
    npts, dim = size(w.points)
    domain = support(w.knots)
    print(io, "TensorProductWeights{$T}(number_pts=$npts, dim=$dim, support=$(domain))")
end

# todo: update doc strings, test and documentation !

# function Base.show(io::IO, ::MIME"text/plain", w::TensorProductWeights)
#     npts, dim = size(w.points)
#     weight_min = minimum(w.weights)
#     weight_max = maximum(w.weights)
#     weight_sum = sum(w.weights)

#     println(io, "TensorProductWeights:")
#     println(io, "  Number of points: $npts")
#     println(io, "  Dimensions: $dim")
#     println(io, "  Quadrature type: Tensor product Gauss-Hermite")
#     println(io, "  Reference measure: Standard Gaussian")
#     println(io, "  Weight range: [$weight_min, $weight_max]")
# end


# Display methods for MonteCarloWeights
function Base.show(io::IO, w::MonteCarloWeights)
    npts, dim = size(w.points)
    print(io, "MonteCarloWeights($npts points, $dim dimensions)")
end

function Base.show(io::IO, ::MIME"text/plain", w::MonteCarloWeights)
    npts, dim = size(w.points)
    weight_value = w.weights[1]  # All weights are the same for Monte Carlo

    println(io, "MonteCarloWeights:")
    println(io, "  Number of points: $npts")
    println(io, "  Dimensions: $dim")
    println(io, "  Sampling type: Random (Gaussian)")
    println(io, "  Reference measure: Standard Gaussian")
    println(io, "  Weight (uniform): $weight_value")
end

# Display methods for LatinHypercubeWeights
function Base.show(io::IO, w::LatinHypercubeWeights)
    npts, dim = size(w.points)
    print(io, "LatinHypercubeWeights($npts points, $dim dimensions)")
end

function Base.show(io::IO, ::MIME"text/plain", w::LatinHypercubeWeights)
    npts, dim = size(w.points)
    weight_value = w.weights[1]  # All weights are the same for Latin Hypercube

    println(io, "LatinHypercubeWeights:")
    println(io, "  Number of points: $npts")
    println(io, "  Dimensions: $dim")
    println(io, "  Sampling type: Latin Hypercube")
    println(io, "  Reference measure: Standard Gaussian (via inverse CDF)")
    println(io, "  Weight (uniform): $weight_value")
end

function Base.show(io::IO, w::SparseSmolyakWeights)
    npts, dim = size(w.points)
    print(io, "SparseSmolyakWeights($npts points, $dim dimensions)")
end

function Base.show(io::IO, ::MIME"text/plain", w::SparseSmolyakWeights)
    npts, dim = size(w.points)
    weight_min = isempty(w.weights) ? 0.0 : minimum(w.weights)
    weight_max = isempty(w.weights) ? 0.0 : maximum(w.weights)
    weight_sum = sum(w.weights)

    println(io, "SparseSmolyakWeights:")
    println(io, "  Number of points: $npts")
    println(io, "  Dimensions: $dim")
    println(io, "  Quadrature type: Sparse Smolyak (Gauss-Hermite)")
    println(io, "  Reference measure: Standard Gaussian")
    println(io, "  Weight range: [$weight_min, $weight_max]")
end

function numberdimensions(quad::AbstractQuadratureWeights)
    return size(quad.points, 2)
end
