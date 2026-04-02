# Implementation of various quadrature rules for numerical integration

"""
    GaussHermiteWeights

Tensor product Gauss-Hermite quadrature for numerical integration with Gaussian
reference measure. Uses `numberpoints` points per dimension, resulting in
`numberpoints^dimension` total quadrature points.

# Fields
- `points::Matrix{Float64}`: Quadrature points
- `weights::Vector{Float64}`: Quadrature weights

# Constructors
- `GaussHermiteWeights(numberpoints::Int64, dimension::Int64)`: Construct weights for number of points per dimension and `dimension`.
- `GaussHermiteWeights(numberpoints::Int64, map::AbstractTransportMap)`: Get number of dimensions from `map` and construct a standard Gaussian Gauss-Hermite rule.
"""
struct GaussHermiteWeights <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}

    function GaussHermiteWeights(numberpoints::Int64, dimension::Int64)
        points, weights = gausshermite_weights(numberpoints, dimension)
        return new(points, weights)
    end

    function GaussHermiteWeights(numberpoints::Int64, map::AbstractTransportMap)
        # @warn "Using standard Gauss-Hermite quadrature with standard Gaussian reference density."
        # Generate Gauss-Hermite points in the reference space
        points, weights = gausshermite_weights(numberpoints, numberdimensions(map))
        return new(points, weights)
    end
end

function gausshermite_weights(numberpoints::Int64, dimension::Int64)
    # Tensor product Gauss-Hermite quadrature
    x1d, w1d = gausshermite(numberpoints; normalize=true)

    # Generate tensor product indices
    indices = collect(Iterators.product(ntuple(_ -> 1:numberpoints, dimension)...))

    # Allocate arrays for points and weights
    points = Matrix{Float64}(undef, length(indices), dimension)
    weights = Vector{Float64}(undef, length(indices))

    for (k, idx) in enumerate(indices)
        points[k, :] = [x1d[i] for i in idx]
        weights[k] = prod(w1d[i] for i in idx)
    end

    return points, weights
end

"""
    GaussLegendreWeights

Tensor product Gauss-Legendre quadrature for numerical integration with Uniform
reference measure. Automatically handles both U[-1,1] and U[0,1] reference distributions.

# Fields
- `points::Matrix{Float64}`: Quadrature points
- `weights::Vector{Float64}`: Quadrature weights

# Constructors
- `GaussLegendreWeights(numberpoints::Int64, dimension::Int64)`: Construct for U[-1,1] (default).
- `GaussLegendreWeights(numberpoints::Int64, dimension::Int64, reference::Uniform)`: Construct for specific Uniform distribution.
- `GaussLegendreWeights(numberpoints::Int64, map::AbstractTransportMap)`: Auto-detect from map's reference (must be `Uniform`).
"""
struct GaussLegendreWeights <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}

    function GaussLegendreWeights(numberpoints::Int64, dimension::Int64)
        # Default: U[-1,1]
        points, weights = gausslegendre_weights(numberpoints, dimension, Uniform(-1, 1))
        return new(points, weights)
    end

    function GaussLegendreWeights(numberpoints::Int64, dimension::Int64, reference::Uniform)
        points, weights = gausslegendre_weights(numberpoints, dimension, reference)
        return new(points, weights)
    end

    function GaussLegendreWeights(numberpoints::Int64, map::AbstractTransportMap)
        # Extract reference distribution from map
        ref_dist = map.reference.densitytype

        if !(ref_dist isa Uniform)
            error("GaussLegendreWeights requires Uniform reference distribution, got $(typeof(ref_dist))")
        end

        points, weights = gausslegendre_weights(numberpoints, numberdimensions(map), ref_dist)
        return new(points, weights)
    end
end

function gausslegendre_weights(numberpoints::Int64, dimension::Int64, reference::Uniform)
    # Tensor product Gauss-Legendre quadrature on [-1, 1]
    x1d, w1d = gausslegendre(numberpoints)

    # Get the bounds of the reference distribution
    a, b = reference.a, reference.b

    # Transform from [-1, 1] to [a, b]: x_transformed = (b-a)/2 * x + (a+b)/2
    # Weight scaling: w_transformed = (b-a)/2 * w
    scale = (b - a) / 2
    shift = (a + b) / 2

    # Generate tensor product indices
    indices = collect(Iterators.product(ntuple(_ -> 1:numberpoints, dimension)...))

    # Allocate arrays for points and weights
    points = Matrix{Float64}(undef, length(indices), dimension)
    weights = Vector{Float64}(undef, length(indices))

    for (k, idx) in enumerate(indices)
        # Transform each point from [-1,1] to [a,b]
        points[k, :] = [scale * x1d[i] + shift for i in idx]
        # Scale weights appropriately (scale^dimension for d dimensions)
        weights[k] = (scale^dimension) * prod(w1d[i] for i in idx)
    end

    return points, weights
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

"""
    SparseSmolyakWeights

Sparse Smolyak quadrature using Gauss-Hermite rules. Reduces the curse of
dimensionality by using a sparse grid construction. The `level` parameter
controls accuracy (higher level = more points and higher accuracy).

# Fields
- `points::Matrix{Float64}`: Quadrature points (sparse grid)
- `weights::Vector{Float64}`: Quadrature weights

# Constructors
- `SparseSmolyakWeights(level::Int64, dimension::Int64)`: Construct sparse Smolyak grid with specified `level` and `dimension`.
- `SparseSmolyakWeights(level::Int64, map::AbstractTransportMap)`: Get number of dimensions from `map` and construct a standard Gaussian sparse Smolyak rule.
"""
struct SparseSmolyakWeights <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}

    function SparseSmolyakWeights(level::Int64, dimension::Int64)
        points, weights = hermite_smolyak_points(dimension, level)
        return new(points, weights)
    end

    function SparseSmolyakWeights(level::Int64, map::AbstractTransportMap)
        # @warn "Using Smolyak sparse Gauss-Hermite quadrature with standard Gaussian reference density."
        points, weights = hermite_smolyak_points(numberdimensions(map), level)
        return new(points, weights)
    end
end

# Display methods for GaussHermiteWeights
function Base.show(io::IO, w::GaussHermiteWeights)
    npts, dim = size(w.points)
    print(io, "GaussHermiteWeights($npts points, $dim dimensions)")
end

function Base.show(io::IO, ::MIME"text/plain", w::GaussHermiteWeights)
    npts, dim = size(w.points)
    weight_min = minimum(w.weights)
    weight_max = maximum(w.weights)
    weight_sum = sum(w.weights)

    println(io, "GaussHermiteWeights:")
    println(io, "  Number of points: $npts")
    println(io, "  Dimensions: $dim")
    println(io, "  Quadrature type: Tensor product Gauss-Hermite")
    println(io, "  Reference measure: Standard Gaussian")
    println(io, "  Weight range: [$weight_min, $weight_max]")
end

# Display methods for GaussLegendreWeights
function Base.show(io::IO, w::GaussLegendreWeights)
    npts, dim = size(w.points)
    print(io, "GaussLegendreWeights($npts points, $dim dimensions)")
end

function Base.show(io::IO, ::MIME"text/plain", w::GaussLegendreWeights)
    npts, dim = size(w.points)
    weight_min = minimum(w.weights)
    weight_max = maximum(w.weights)
    weight_sum = sum(w.weights)

    # Detect domain from points
    point_min = minimum(w.points)
    point_max = maximum(w.points)
    domain_str = "[$point_min, $point_max]^$dim"

    println(io, "GaussLegendreWeights:")
    println(io, "  Number of points: $npts")
    println(io, "  Dimensions: $dim")
    println(io, "  Quadrature type: Tensor product Gauss-Legendre")
    println(io, "  Reference domain: $domain_str")
    println(io, "  Weight range: [$weight_min, $weight_max]")
    println(io, "  Sum of weights: $weight_sum")
end

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
