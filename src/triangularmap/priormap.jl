"""
    PriorMap(marginals::Vector{<:ContinuousUnivariateDistribution}; reference = Normal())

Fixed diagonal transport from a product of the continuous univariate `reference`
distribution to independent continuous prior marginals. The reference defaults
to `Normal()`; `Uniform(0, 1)` and `Uniform(-1, 1)` are also supported.
`evaluate(P, z)[i] = quantile(P.marginals[i], cdf(P.reference.densitytype, z[i]))`;
`inverse(P, x)` transforms physical coordinates back to the reference.
Vectors are single points; matrices contain observations in rows.

The marginal vector must be nonempty and is copied on construction. Marginals
must implement the Distributions.jl density, CDF, and quantile interfaces.
Discrete distributions and dependent priors are not
supported. Log-tail probabilities are used to reduce tail cancellation; finite
precision can still round extreme quantiles to support boundaries. Inversion
at a boundary may return infinity; transformation inputs outside their source
support throw `DomainError`. Derivatives are defined in the interior of the support. A
target-directed composition assigns zero density outside the prior support
and at boundaries whose inverse coordinates are infinite.

`jacobian` returns the forward **determinant**, following the map convention.
`logabsdetjacobian` sums log-density ratios. ForwardDiff input derivatives use
analytical scalar transformation derivatives, without differentiating quantiles
or CDFs. Distribution parameters are fixed.

After fitting `T` to the posterior in prior-reference coordinates, use
`ComposedMap(T, P; forwarddirection=:target)` to obtain a map into physical
posterior coordinates. The output of `T` must lie in the support of
`P.reference.densitytype`; polynomial corrections do not automatically preserve
bounded supports. `PriorMap` is a transport layer, not a distribution.
"""
struct PriorMap{D <: Distributions.ContinuousUnivariateDistribution, R <: Distributions.ContinuousUnivariateDistribution} <: AbstractTriangularMap
    marginals::Vector{D}
    reference::MapReferenceDensity
    forwarddirection::Symbol

    function PriorMap(marginals::Vector{<:Distributions.ContinuousUnivariateDistribution}; reference = Normal())
        isempty(marginals) && throw(ArgumentError("PriorMap requires at least one marginal"))
        reference isa Distributions.ContinuousUnivariateDistribution ||
            throw(ArgumentError("reference must be a continuous univariate distribution"))
        return new{eltype(marginals), typeof(reference)}(copy(marginals), MapReferenceDensity(reference), :target)
    end
end

numberdimensions(P::PriorMap) = length(P.marginals)

function _check_prior_input(P::PriorMap, x)
    n = x isa AbstractVector ? length(x) : size(x, 2)
    n == numberdimensions(P) || throw(DimensionMismatch("Expected $(numberdimensions(P)) coordinates, got $n"))
    return nothing
end

# Use the smaller probability tail; the split depends on the source CDF,
# rather than on a particular reference distribution's center.
function _prior_transform_value(to, from, x)
    lp = logcdf(from, x)
    return lp <= -log(2) ? invlogcdf(to, lp) : invlogccdf(to, logccdf(from, x))
end
_prior_transform_value(to::Normal, from::Normal, x) = mean(to) + std(to) * ((x - mean(from)) / std(from))
_prior_transform_value(to::Uniform, from::Uniform, x) = minimum(to) + (maximum(to) - minimum(to)) * ((x - minimum(from)) / (maximum(from) - minimum(from)))

function _prior_transform(to, from, x::Real)
    insupport(from, x) || throw(DomainError(x, "Coordinate is outside its source distribution support"))
    return _prior_transform_value(to, from, x)
end
function _prior_transform(to, from, x::ForwardDiff.Dual{Tag}) where {Tag}
    v = ForwardDiff.value(x)
    y = _prior_transform(to, from, v)
    derivative = exp(_prior_logratio(to, from, v, y))
    return ForwardDiff.Dual{Tag}(y, derivative * ForwardDiff.partials(x))
end

_prior_logratio(to, from, x, y) = logpdf(from, x) - logpdf(to, y)
_prior_logratio(to::Normal, from::Normal, x, y) = log(std(to)) - log(std(from))
_prior_logratio(to::Uniform, from::Uniform, x, y) = log(maximum(to) - minimum(to)) - log(maximum(from) - minimum(from))

function evaluate(P::PriorMap, z::AbstractVector{<:Real})
    _check_prior_input(P, z)
    return [_prior_transform(d, P.reference.densitytype, z[i]) for (i, d) in enumerate(P.marginals)]
end
function inverse(P::PriorMap, x::AbstractVector{<:Real})
    _check_prior_input(P, x)
    return [_prior_transform(P.reference.densitytype, d, x[i]) for (i, d) in enumerate(P.marginals)]
end
function evaluate(P::PriorMap, Z::AbstractMatrix{<:Real})
    _check_prior_input(P, Z)
    return [_prior_transform(P.marginals[j], P.reference.densitytype, Z[i, j]) for i in axes(Z, 1), j in axes(Z, 2)]
end
function inverse(P::PriorMap, X::AbstractMatrix{<:Real})
    _check_prior_input(P, X)
    return [_prior_transform(P.reference.densitytype, P.marginals[j], X[i, j]) for i in axes(X, 1), j in axes(X, 2)]
end

function _prior_logderivative(d, reference, z)
    x = _prior_transform(d, reference, z)
    return _prior_logratio(d, reference, z, x)
end
function logabsdetjacobian(P::PriorMap, z::AbstractVector{<:Real})
    _check_prior_input(P, z)
    return sum(_prior_logderivative(d, P.reference.densitytype, z[i]) for (i, d) in enumerate(P.marginals))
end
function logabsdetjacobian(P::PriorMap, Z::AbstractMatrix{<:Real})
    _check_prior_input(P, Z)
    return [logabsdetjacobian(P, z) for z in eachrow(Z)]
end
jacobian(P::PriorMap, z::AbstractVector{<:Real}) = exp(logabsdetjacobian(P, z))
jacobian(P::PriorMap, Z::AbstractMatrix{<:Real}) = exp.(logabsdetjacobian(P, Z))
function inverse_jacobian(P::PriorMap, x::AbstractVector{<:Real})
    z = inverse(P, x)
    return exp(sum(_prior_logratio(P.reference.densitytype, d, x[i], z[i]) for (i, d) in enumerate(P.marginals)))
end
function inverse_jacobian(P::PriorMap, X::AbstractMatrix{<:Real})
    _check_prior_input(P, X)
    return [inverse_jacobian(P, x) for x in eachrow(X)]
end

function _inverse_logabsdet(P::PriorMap, x)
    insupport(P, x) || return nothing, -Inf
    z = inverse(P, x)
    # At a support boundary the inverse can be infinite. A density's value
    # there is immaterial to the measure; use zero rather than an Inf-Inf NaN.
    all(isfinite, z) || return nothing, -Inf
    return z, logpdf(P, x) - logpdf(P.reference, z)
end

function Distributions.insupport(P::PriorMap, x::AbstractVector{<:Real})
    _check_prior_input(P, x)
    return all(insupport(d, x[i]) for (i, d) in enumerate(P.marginals))
end
function logpdf(P::PriorMap, x::AbstractVector{<:Real})
    _check_prior_input(P, x)
    return sum(logpdf(d, x[i]) for (i, d) in enumerate(P.marginals))
end
function logpdf(P::PriorMap, X::AbstractMatrix{<:Real})
    _check_prior_input(P, X)
    return [logpdf(P, x) for x in eachrow(X)]
end
pdf(P::PriorMap, x::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) = exp.(logpdf(P, x))
pullback(P::PriorMap, x::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) = pdf(P, x)
(P::PriorMap)(z::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) = evaluate(P, z)

function Base.show(io::IO, P::PriorMap)
    print(io, "PriorMap(")
    print(io, "[")
    for (i, marginal) in enumerate(P.marginals)
        i > 1 && print(io, ", ")
        show(io, marginal)
    end
    print(io, "]")
    print(io, "; reference=")
    show(io, P.reference.densitytype)
    return print(io, ")")
end
