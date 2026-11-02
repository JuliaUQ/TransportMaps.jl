"""
    TransportMapDistribution(map::M) where {M <: AbstractTransportMap}

Represent a fitted `PolynomialMap`, `ComposedMap`, `LinearMap`, or `LaplaceMap`
as a [`Distributions.ContinuousMultivariateDistribution`](https://juliastats.org/Distributions.jl/stable/multivariate/).
Standalone linear and Laplace maps use a product standard-normal reference and map target to
reference, so their inverse generates the corresponding Gaussian samples. Compositions
inherit the polynomial layer's reference and the direction of the whole map.

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

    logdet = [logabsdetjacobian(tm, point) for point in eachrow(points)]
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
