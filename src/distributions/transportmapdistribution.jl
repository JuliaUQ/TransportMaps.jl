"""
    TransportMapDistribution(map::M) where {M <: AbstractTransportMap}

Represent a fitted transport map as a [`Distributions.ContinuousMultivariateDistribution`](https://juliastats.org/Distributions.jl/stable/multivariate/).

A `TransportMapDistribution` wraps a trained transport map (`PolynomialMap`, `ComposedMap`,
`LinearMap`, or `LaplaceMap`) to create a probability distribution that can be sampled from,
evaluated, and analyzed. The distribution represents the pushforward of a reference distribution
through the transport map.

## Map Types

- **`PolynomialMap`**: A polynomial transport map with adjustable reference distribution and direction.
- **`ComposedMap`**: A composition of layers (polynomial, linear, Laplace, or custom invertible layers),
  each with optional `MarginalMap` wrappers for dimension-by-dimension flexibility.
- **`LinearMap`** (standalone): Uses a product standard-normal reference and maps target to reference,
  so its inverse generates Gaussian samples with empirical mean/covariance.
- **`LaplaceMap`** (standalone): Similar to `LinearMap`, but based on the Laplace approximation with
  a lower-triangular Cholesky factor.

## Key Features

- **Sampling**: Generate samples from the approximated target distribution via `rand(d)`.
- **Density evaluation**: Compute log-density with `logpdf(d, x)` and density with `pdf(d, x)`.
- **Moments**: Compute mean, standard deviation, covariance matrix, correlation matrix, and entropy
  via numerical quadrature (default: level-3 sparse Smolyak), with exact computation for linear and
  Laplace maps.
- **Flexibility**: Standalone maps use a product standard-normal reference; composed maps can specify
  their own reference and direction.

## Construction

Use one of the three main constructors:
1. From samples: `TransportMapDistribution(samples, order, adaptive)`
2. From target density: `TransportMapDistribution(target, quadrature, order, adaptive)`
3. From a pre-fitted map: `TransportMapDistribution(map)`

# Arguments
- `map::M`: A fitted transport map implementing `evaluate`, `inverse`, and `logabsdetjacobian` methods.
  The map must have positive dimensionality.

# Example
```julia
# Create distribution from a fitted map
map = PolynomialMap(2, 2)  # initialize and fit map, then optimize!() etc.
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

# Exceptions
- `ArgumentError`: Raised if the map is zero-dimensional or not a supported type.
- `DimensionMismatch`: Raised if composed map layers have inconsistent dimensions.
"""
struct TransportMapDistribution{M <: AbstractTransportMap} <: Distributions.ContinuousMultivariateDistribution
    map::M

    function TransportMapDistribution(map::M) where {M <: AbstractTransportMap}
        numberdimensions(map) > 0 || throw(ArgumentError("A distribution requires a dimensioned map; use LinearMap(n) for a standalone identity"))
        _validate_distribution_map(map)
        return new{M}(map)
    end
end

"""
    TransportMapDistribution(samples::Matrix{Float64}, order::Union{Int, Vector{Int}}, adaptive::Bool = true; reference::UnivariateDistribution = Normal(), rectifier::AbstractRectifierFunction = Softplus(), use_linear_map::Bool = true, k_folds::Int = 5)

Construct a `TransportMapDistribution` by learning a transport map from sample data.

This constructor fits a transport map to approximate the target distribution represented by
the provided samples. It automatically constructs and optimizes a `PolynomialMap` (optionally
composed with a `LinearMap`) to map from a reference distribution (default: standard normal)
to match the empirical distribution of the samples.

# Arguments
- `samples::Matrix{Float64}`: Sample data where each row is a sample and each column is a
  dimension. Shape: (n_samples, n_dimensions).
- `order::Union{Int, Vector{Int}}`: Maximum polynomial order for each dimension. For adaptive
  maps, provide a vector of orders (one per dimension); for fixed maps, provide a single integer.
- `adaptive::Bool = true`: If `true`, use adaptive map construction with cross-validation to
  select optimal polynomial terms. If `false`, construct a fixed-order map.

# Keyword Arguments
- `reference::UnivariateDistribution = Normal()`: Reference distribution for the map. Must be
  a continuous univariate distribution.
- `rectifier::AbstractRectifierFunction = Softplus()`: Rectifier function for nonlinear components.
- `use_linear_map::Bool = true`: If `true`, first apply a linear transformation (computed from
  sample statistics) before polynomial refinement.
- `k_folds::Int = 5`: Number of cross-validation folds for adaptive map selection (only used
  if `adaptive=true`).

# Example
```julia
# Fit a fixed-order map to samples
samples = randn(1000, 2)
tmd = TransportMapDistribution(samples, 5, false; k_folds = 5)

# Fit an adaptive map with dimension-specific orders
tmd_adaptive = TransportMapDistribution(samples, [10, 10], true; use_linear_map = true)
```

# Returns
A fitted `TransportMapDistribution` that can be used for sampling, density evaluation, and
computing moments (mean, covariance, entropy, etc.).

See also: [`TransportMapDistribution(::Function, ::AbstractQuadratureWeights, ::Int, ::Bool)`](@ref)
"""
function TransportMapDistribution(samples::Matrix{Float64}, order::Union{Int, Vector{Int}}, adaptive::Bool = true; reference::UnivariateDistribution = Normal(), rectifier::AbstractRectifierFunction = Softplus(), use_linear_map::Bool = true, k_folds::Int = 5)

    @assert adaptive || isa(order, Int) "Fixed map construction requires a single integer for `order`."

    ndim = size(samples, 2) #!TODO: Check which dimension

    if adaptive
        if isa(order, Int) && ndim > 1
            order = fill(order, size(samples, 2))
            @warn "Adaptive map construction requires a vector of maximum components for each dimension; using $order for all dimensions."
        elseif isa(order, Int) && ndim == 1
            order = [order]
        end
        return _adaptive_distribution_from_samples(samples, order, rectifier, k_folds, use_linear_map)
    else
        return _basic_distribution_from_samples(samples, order, reference, rectifier, use_linear_map)
    end
end

"""
    TransportMapDistribution(target::Function, quadrature::AbstractQuadratureWeights, order::Int, adaptive::Bool = true; basis::AbstractPolynomialBasis = LinearizedHermiteBasis(), rectifier::AbstractRectifierFunction = Softplus(), validation::AbstractQuadratureWeights = quadrature)

Construct a `TransportMapDistribution` by learning a transport map from a target density function.

This constructor fits a transport map to approximate a target probability density by optimizing
the map parameters to match the target through numerical quadrature. The map is constructed
as a `PolynomialMap` (optionally with adaptive term selection) that maps from a reference
distribution (standard normal) to the target.

# Arguments
- `target::Function`: A function computing `log(π(x))` where π is the target density. Must accept
  a vector and return a real value (the log-density).
- `quadrature::AbstractQuadratureWeights`: Quadrature rule for numerical integration during
  optimization (e.g., `SparseSmolyakWeights(3, 2)` for 3D with level 3).
- `order::Int`: Maximum polynomial order for the transport map.
- `adaptive::Bool = true`: If `true`, use adaptive map construction with cross-validation to
  select optimal polynomial terms. If `false`, construct a fixed-order map.

# Keyword Arguments
- `rectifier::AbstractRectifierFunction = Softplus()`: Rectifier function for nonlinear components.
- `validation::AbstractQuadratureWeights = quadrature`: Quadrature rule for validation during
  adaptive optimization. Defaults to the same rule as `quadrature`.

# Example
```julia
# Define target log-density
logtarget(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^3)

# Create quadrature rule
quadr = SparseSmolyakWeights(3, 2)

# Fit a fixed-order map
tmd = TransportMapDistribution(logtarget, quadr, 5, false)

# Fit an adaptive map
tmd_adaptive = TransportMapDistribution(logtarget, quadr, 27, true)
```

# Returns
A fitted `TransportMapDistribution` that can be used for sampling, density evaluation, and
computing moments (mean, covariance, entropy, etc.).

See also: [`TransportMapDistribution(::Matrix{Float64}, ::Union{Int, Vector{Int}}, ::Bool)`](@ref)
"""
function TransportMapDistribution(target::Function, quadrature::AbstractQuadratureWeights, order::Int, adaptive::Bool = true; rectifier::AbstractRectifierFunction = Softplus(), validation::AbstractQuadratureWeights = quadrature, reference::UnivariateDistribution = Normal())

    if adaptive
        return _adaptive_distribution_from_target(target, quadrature, order, rectifier, validation)
    else
        return _basic_distribution_from_target(target, quadrature, order, reference, rectifier)
    end

end

# basic map from samples
function _basic_distribution_from_samples(samples::Matrix{Float64}, order::Int, reference::UnivariateDistribution, rectifier::AbstractRectifierFunction, use_linear_map::Bool)

    ndim = size(samples, 2) #!TODO: Check which dimension

    if use_linear_map
        lmap = LinearMap(samples)
        normal_map = PolynomialMap(ndim, order, reference, rectifier)
        normal_result = optimize!(normal_map, samples, lmap)
        return TransportMapDistribution(ComposedMap(lmap, normal_map))
    else
        normal_map = PolynomialMap(ndim, order, reference, rectifier)
        normal_result = optimize!(normal_map, samples)
        return TransportMapDistribution(normal_map)
    end
end

# basic map from target
function _basic_distribution_from_target(target::Function, quadrature::AbstractQuadratureWeights, order::Int, reference::UnivariateDistribution, rectifier::AbstractRectifierFunction)

    ndim = size(quadrature.points, 2)

    maptarget = MapTargetDensity(target)
    normal_map = PolynomialMap(ndim, order, reference, rectifier)
    normal_result = optimize!(normal_map, maptarget, quadrature)

    return TransportMapDistribution(normal_map)
end

# adaptive map from samples
function _adaptive_distribution_from_samples(samples::Matrix{Float64}, max_components::Vector{Int}, rectifier::AbstractRectifierFunction, k_folds::Int, use_linear_map::Bool)

    if use_linear_map
        lmap = LinearMap(samples)
        M, results, selected_terms, selected_folds = optimize_adaptive_transportmap(
            samples, max_components, k_folds, lmap, rectifier
        )
        return TransportMapDistribution(M)
    else
        lmap = LinearMap(size(samples, 2)) #!TODO: Check which dimension
        M, results, selected_terms, selected_folds = optimize_adaptive_transportmap(
            samples, max_components, k_folds, lmap, rectifier
        )
        return TransportMapDistribution(M)
    end
end

function _adaptive_distribution_from_target(target::Function, quadrature::AbstractQuadratureWeights, n_terms::Int, rectifier::AbstractRectifierFunction, validation::AbstractQuadratureWeights)

    maptarget = MapTargetDensity(target)

    T, hist = optimize_adaptive_transportmap(
        maptarget, quadrature, n_terms;
        rectifier = rectifier,
        validation = validation
    )

    return TransportMapDistribution(T)
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
    reference = map.reference
    if reference isa MapReferenceDensity
        reference.densitytype isa Distributions.ContinuousUnivariateDistribution ||
            throw(ArgumentError("TransportMapDistribution requires a continuous reference"))
    elseif reference isa Distributions.ContinuousMultivariateDistribution
        length(reference) == numberdimensions(map) ||
            throw(DimensionMismatch("Reference and map dimensions must agree"))
    else
        throw(ArgumentError("TransportMapDistribution requires a sampleable continuous product or multivariate reference"))
    end
    for layer in map.layers
        numberdimensions(layer) == numberdimensions(map) ||
            (layer isa LinearMap && numberdimensions(layer) == 0) ||
            throw(DimensionMismatch("Composed map dimensions must agree"))
        _validate_distribution_layer(layer, numberdimensions(map))
    end
    return nothing
end

_validate_distribution_layer(map::Union{PolynomialMap, LinearMap, LaplaceMap}, dim) =
    _validate_distribution_map(map)

function _validate_distribution_layer(map::AbstractTransportMap, dim)
    point = zeros(dim)
    applicable(evaluate, map, point) && applicable(inverse, map, point) ||
        throw(ArgumentError("Distribution layers must implement evaluate and inverse"))
    return nothing
end

function Base.show(io::IO, d::TransportMapDistribution)
    print(io, "TransportMapDistribution(")
    print(io, "map=$(d.map)")
    return print(io, ")")
end

"""
    length(d::TransportMapDistribution)

Return the sample dimension of distribution `d`.
"""
Base.length(d::TransportMapDistribution) = numberdimensions(d.map)
Distributions.sampler(d::TransportMapDistribution) = d
Base.eltype(::Type{<:TransportMapDistribution}) = Float64

## Helper functions
function _reference_distribution(tm::AbstractTransportMap)
    return product_distribution(fill(_reference_marginal(tm), numberdimensions(tm)))
end

function _reference_distribution(tm::ComposedMap)
    tm.reference isa MapReferenceDensity || return tm.reference
    return product_distribution(fill(tm.reference.densitytype, numberdimensions(tm)))
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
    # Transform points from reference to target space
    X = _from_reference(tm, quad.points)
    # Compute nth power element-wise
    Xn = X .^ n
    # Return weighted sum (moment)
    return Xn' * quad.weights
end

function _centered_map_values(tm::AbstractTransportMap, quad::AbstractQuadratureWeights)
    # Transform points from reference to target space
    X = _from_reference(tm, quad.points)
    # Compute mean using weighted sum
    μ = X' * quad.weights
    # Center the values (subtract mean from each point)
    # Use @. for type-stable broadcasting
    return @. X - μ'
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
"""
    mean(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))

Return the mean vector of the fitted distribution. Let ``\\rho`` be the joint
reference density and ``F`` the transformation from reference to target:

```math
F(z) = \\begin{cases}
M(z), & \\text{if the map direction is } \\mathtt{:target}, \\\\
M^{-1}(z), & \\text{if the map direction is } \\mathtt{:reference}.
\\end{cases}
```

For reference-space quadrature points ``z_q`` and weights ``w_q``, compute

```math
\\mu = \\int F(z)\\rho(z)\\,\\mathrm{d}z
\\approx \\widehat\\mu = \\sum_{q=1}^{N} w_q F(z_q).
```

Polynomial and composed maps use a level-three sparse Smolyak rule by default.
Standalone linear and Laplace maps use exact Gaussian statistics and ignore
`quad`: their reference-to-target transformation is ``F(z)=m+Az``, giving
``\\mu=m``. For `LinearMap`, ``m`` is `map.μ` and ``A=\\operatorname{diag}(\\sigma)``;
for `LaplaceMap`, ``m`` is `map.mode` and ``A`` is `map.chol`.
"""
function mean(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    return _nth_moment(d.map, 1, quad)
end

"""
    var(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))

Return the vector of marginal variances of the fitted distribution. With the
reference-to-target transformation ``F`` and quadrature mean ``\\widehat\\mu``
defined in [`mean`](@ref), compute each component using centered values:

```math
v_i = \\int \\bigl(F_i(z)-\\mu_i\\bigr)^2\\rho(z)\\,\\mathrm{d}z
\\approx \\sum_{q=1}^{N} w_q\\bigl(F_i(z_q)-\\widehat\\mu_i\\bigr)^2.
```

Polynomial and composed maps use reference-space quadrature, with a level-three
sparse Smolyak rule by default. Standalone linear and Laplace maps ignore `quad`
and compute the exact Gaussian variances ``v_i=(AA^{\\mathsf T})_{ii}``, where
``A`` is defined in [`mean`](@ref).
"""
function var(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    centered = _centered_map_values(d.map, quad)
    return (centered .^ 2)' * quad.weights
end

"""
    std(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))

Return the vector of marginal standard deviations of the fitted distribution:

```math
\\sigma_i = \\sqrt{v_i}
\\approx \\sqrt{\\sum_{q=1}^{N} w_q\\bigl(F_i(z_q)-\\widehat\\mu_i\\bigr)^2}.
```

Here ``F`` and ``\\widehat\\mu`` are defined in [`mean`](@ref). Polynomial and
composed maps use the variances from [`var`](@ref), with a level-three sparse
Smolyak rule by default. Standalone linear and Laplace maps ignore `quad` and
compute the exact Gaussian standard deviations ``\\sigma_i=\\lVert A_{i,:}\\rVert_2``.
"""
function std(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    return sqrt.(var(d; quad = quad))
end

"""
    cov(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))

Return the covariance matrix of the fitted distribution. With ``F`` and
``\\widehat\\mu`` defined in [`mean`](@ref), compute

```math
\\begin{aligned}
\\Sigma &= \\int \\bigl(F(z)-\\mu\\bigr)\\bigl(F(z)-\\mu\\bigr)^{\\mathsf T}
    \\rho(z)\\,\\mathrm{d}z, \\\\
\\widehat\\Sigma &= \\sum_{q=1}^{N} w_q
    \\bigl(F(z_q)-\\widehat\\mu\\bigr)\\bigl(F(z_q)-\\widehat\\mu\\bigr)^{\\mathsf T}
    \\approx \\Sigma.
\\end{aligned}
```

Polynomial and composed maps use reference-space quadrature, with a level-three
sparse Smolyak rule by default. Standalone linear and Laplace maps ignore `quad`
and compute the exact Gaussian covariance ``\\Sigma=AA^{\\mathsf T}``, where
``A`` is defined in [`mean`](@ref).
"""
function cov(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    centered = _centered_map_values(d.map, quad)
    return centered' * (quad.weights .* centered)
end

"""
    cor(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))

Return the correlation matrix of the fitted distribution by normalizing its
covariance matrix:

```math
R_{ij} = \\frac{\\Sigma_{ij}}{\\sqrt{\\Sigma_{ii}\\Sigma_{jj}}}.
```

Polynomial and composed maps compute [`cov`](@ref) once using reference-space
quadrature, with a level-three sparse Smolyak rule by default. Standalone linear
and Laplace maps use exact Gaussian correlations and ignore `quad`. Linear maps
give ``R=I``; Laplace maps normalize the rows of `map.chol` before forming their
pairwise inner products.
"""
function cor(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    C = cov(d; quad = quad)
    σ = sqrt.(diag(C))
    return C ./ (σ .* σ')
end

"""
    entropy(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))

Return the differential entropy of the fitted distribution. With the
reference-to-target transformation ``F`` defined in [`mean`](@ref), compute

```math
H(\\pi) = H(\\rho) + \\int \\log\\lvert\\det\\nabla F(z)\\rvert\\rho(z)\\,\\mathrm{d}z
\\approx H(\\rho) + \\sum_{q=1}^{N} w_q\\log\\lvert\\det\\nabla F(z_q)\\rvert.
```

The log-Jacobian correction follows the direction of the whole map:

```math
\\log\\lvert\\det\\nabla F(z)\\rvert = \\begin{cases}
\\log\\lvert\\det\\nabla M(z)\\rvert,
    & \\text{for } \\mathtt{:target}, \\\\
-\\log\\lvert\\det\\nabla M(F(z))\\rvert,
    & \\text{for } \\mathtt{:reference}.
\\end{cases}
```

Polynomial and composed maps use a level-three sparse Smolyak rule by default.
Standalone linear and Laplace maps ignore `quad` and return the exact Gaussian
entropy for dimension ``n=\\operatorname{length}(d)``:

```math
H(\\pi) = \\frac{n}{2}\\log(2\\pi e) + \\log\\lvert\\det A\\rvert,
```

where ``A`` is defined in [`mean`](@ref). Logarithms are natural, so entropy is
measured in nats.
"""
function entropy(d::TransportMapDistribution; quad::AbstractQuadratureWeights = SparseSmolyakWeights(3, d.map))
    tm = d.map
    points = _map_direction(tm) == :target ? quad.points :
        _from_reference(tm, quad.points)

    # Use matrix version of logabsdetjacobian for better performance
    logdet = logabsdetjacobian(tm, points)
    correction = dot(quad.weights, logdet)
    H_reference = entropy(_reference_distribution(tm))

    return _map_direction(tm) == :target ?
        H_reference + correction :
        H_reference - correction
end

# Dispatch before constructing a default grid. The keyword is accepted for a
# uniform calling convention, but these maps always use exact statistics.
mean(d::TransportMapDistribution{LinearMap}; quad = nothing) = copy(d.map.μ)
mean(d::TransportMapDistribution{LaplaceMap}; quad = nothing) = copy(d.map.mode)
var(d::TransportMapDistribution{LinearMap}; quad = nothing) = d.map.σ .^ 2
var(d::TransportMapDistribution{LaplaceMap}; quad = nothing) = vec(sum(abs2, d.map.chol; dims = 2))
std(d::TransportMapDistribution{LinearMap}; quad = nothing) = abs.(d.map.σ)
std(d::TransportMapDistribution{LaplaceMap}; quad = nothing) = norm.(eachrow(d.map.chol))
cov(d::TransportMapDistribution{LinearMap}; quad = nothing) = Matrix(Diagonal(d.map.σ .^ 2))
cov(d::TransportMapDistribution{LaplaceMap}; quad = nothing) = cov(d.map)
cor(d::TransportMapDistribution{LinearMap}; quad = nothing) = Matrix{Float64}(I, length(d), length(d))

function cor(d::TransportMapDistribution{LaplaceMap}; quad = nothing)
    # Normalize the factor first to avoid squaring extreme scales in covariance.
    normalized = d.map.chol ./ std(d)
    return normalized * normalized'
end

function entropy(d::TransportMapDistribution{LinearMap}; quad = nothing)
    return length(d) * entropy(Normal()) + sum(value -> log(abs(value)), d.map.σ)
end

function entropy(d::TransportMapDistribution{LaplaceMap}; quad = nothing)
    return length(d) * entropy(Normal()) + sum(value -> log(abs(value)), diag(d.map.chol))
end

@doc """
    logpdf(d::TransportMapDistribution, x)

Evaluate the fitted [`logpullback`](@ref) density at `x`.

A vector represents one point; matrix columns represent multiple points.
See [`pdf`](@ref) for evaluation in physical space.
""" Distributions.logpdf(::TransportMapDistribution, ::AbstractVector{<:Real})

@doc """
    pdf(d::TransportMapDistribution, x)

Evaluate the fitted [`pullback`](@ref) density at `x`.

A vector represents one point; matrix columns represent multiple points.
See [`logpullback`](@ref) for details [`logpdf`](@ref) for evaluation in log space.
""" Distributions.pdf(::TransportMapDistribution, ::AbstractVector{<:Real})

## Support
"""
    insupport(d::TransportMapDistribution, x::AbstractVector{<:Real})

Check whether `x` belongs to the support induced by the transport map. The point
must have the distribution's dimension and finite coordinates.
"""
function insupport(d::TransportMapDistribution, x::AbstractVector{<:Real})
    return !isnothing(_reference_point_in_support(d.map, x))
end
