"""
    ComposedMap(maps...; reference = nothing, forwarddirection = :auto)
    ComposedMap(maps::Union{Tuple, AbstractVector}; kwargs...)

Compose maps in evaluation order: `ComposedMap(A, B, C)` applies `C(B(A(x)))`.
`inverse` applies inverse layers in reverse order. Inputs may be vectors or
matrices with observations in rows.

Nested compositions are flattened into the vector `C.layers`. Layer objects are
shared; use `deepcopy` to freeze their parameters. Supply at least one
`AbstractTransportMap`, with matching dimensions (except `LinearMap()`) and
compatible supports.

`forwarddirection` is `:reference` (target -> reference) or `:target`
(reference -> target). `:auto` infers a common direction from supplied maps,
defaulting to `:reference` when none provide one.

`reference` defaults to the last supplied reference for `:reference`, or the
first for `:target`, falling back to a product standard normal. A univariate
distribution defines a product reference. Direction and reference are fixed
at construction.
"""
struct ComposedMap <: AbstractComposedMap
    layers::Vector{AbstractTransportMap}
    reference::Union{AbstractMapDensity, Distributions.Distribution}
    forwarddirection::Symbol

    function ComposedMap(layers::Union{Tuple, AbstractVector}; reference = nothing, forwarddirection::Symbol = :auto)
        isempty(layers) && throw(ArgumentError("ComposedMap requires at least one layer"))
        flat_layers = AbstractTransportMap[]
        for layer in layers
            layer isa AbstractTransportMap ||
                throw(ArgumentError("Every layer must be an AbstractTransportMap"))
            if layer isa ComposedMap
                append!(flat_layers, layer.layers)
            else
                push!(flat_layers, layer)
            end
        end
        all(m -> applicable(numberdimensions, m), flat_layers) ||
            throw(ArgumentError("Every layer must implement numberdimensions"))
        dims = map(numberdimensions, flat_layers)
        all(d -> d isa Integer && d >= 0, dims) ||
            throw(ArgumentError("Layer dimensions must be nonnegative integers"))
        d = maximum(dims)
        all(i -> dims[i] == d || (dims[i] == 0 && flat_layers[i] isa LinearMap), eachindex(flat_layers)) ||
            throw(DimensionMismatch("All layers must have the same dimension"))
        forwarddirection in (:auto, :reference, :target) ||
            throw(ArgumentError("forwarddirection must be :auto, :reference, or :target"))
        forwarddirection === :auto && (forwarddirection = _composition_direction(layers))
        reference === nothing && (reference = _composition_reference(layers, forwarddirection))
        if reference isa Distributions.UnivariateDistribution
            reference = MapReferenceDensity(reference)
        elseif !(reference isa Union{AbstractMapDensity, Distributions.Distribution})
            throw(ArgumentError("reference must be a distribution or AbstractMapDensity"))
        end
        return new(flat_layers, reference, forwarddirection)
    end
end

ComposedMap(maps...; kwargs...) = ComposedMap(maps; kwargs...)

function _composition_direction(layers)
    direction = nothing
    for layer in layers
        hasproperty(layer, :forwarddirection) || continue
        next = layer.forwarddirection
        next in (:reference, :target) || throw(ArgumentError("Invalid layer forwarddirection"))
        direction === nothing || direction === next ||
            throw(ArgumentError("Layers have mixed directions; specify the composition's forwarddirection explicitly"))
        direction = next
    end
    return isnothing(direction) ? :reference : direction
end

function _composition_reference(layers, direction)
    candidates = direction === :reference ? Iterators.reverse(layers) : layers
    for layer in candidates
        hasproperty(layer, :reference) && return layer.reference
    end
    return MapReferenceDensity()
end

_legacy_composition(C::ComposedMap) = length(C.layers) == 2 &&
    C.layers[1] isa AbstractLinearMap && C.layers[2] isa PolynomialMap

function Base.getproperty(C::ComposedMap, name::Symbol)
    if name in (:linearmap, :polynomialmap) && _legacy_composition(C)
        return C.layers[name === :linearmap ? 1 : 2]
    end
    return getfield(C, name)
end

function Base.propertynames(C::ComposedMap, private::Bool = false)
    names = (:layers, :reference, :forwarddirection)
    _legacy_composition(C) && (names = (names..., :linearmap, :polynomialmap))
    return names
end

numberdimensions(C::ComposedMap) = maximum(numberdimensions, C.layers)

function _check_composed_input(C, x)
    d = numberdimensions(C)
    n = x isa AbstractVector ? length(x) : size(x, 2)
    d == 0 || n == d || throw(DimensionMismatch("Input dimension $n does not match map dimension $d"))
    return nothing
end

"""
    evaluate(C::ComposedMap, x)

Apply `C.layers` from first to last, to a vector or a matrix of row-wise points.
"""
function evaluate(C::ComposedMap, x::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}})
    _check_composed_input(C, x)
    isempty(x) && return copy(x)
    for layer in C.layers
        x = evaluate(layer, x)
    end
    return x
end

"""
    inverse(C::ComposedMap, z)

Apply inverse layers from last to first, to a vector or matrix of row-wise points.
"""
function inverse(C::ComposedMap, z::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}})
    _check_composed_input(C, z)
    isempty(z) && return copy(z)
    for layer in Iterators.reverse(C.layers)
        z = inverse(layer, z)
    end
    return z
end

"""
    jacobian(C::ComposedMap, x)

Forward determinant from the chain rule, evaluating each layer's determinant
at that layer's input. Matrix input returns one determinant per row.
"""
function jacobian(C::ComposedMap, x::AbstractVector{<:Real})
    _check_composed_input(C, x)
    value = one(eltype(x))
    layers = C.layers
    for (i, layer) in enumerate(layers)
        value *= jacobian(layer, x)
        i < length(layers) && (x = evaluate(layer, x))
    end
    return value
end

jacobian(C::ComposedMap, X::AbstractMatrix{<:Real}) = (_check_composed_input(C, X); [jacobian(C, x) for x in eachrow(X)])

"""
    logabsdetjacobian(map, x)

Log absolute forward determinant, or one value per matrix row. Compositions
sum layer log-determinants. Custom maps can specialize this for numerical stability.
"""
logabsdetjacobian(M, x::AbstractVector{<:Real}) = log(abs(jacobian(M, x)))
logabsdetjacobian(M, X::AbstractMatrix{<:Real}) = [logabsdetjacobian(M, x) for x in eachrow(X)]

function _evaluate_logabsdet(M, x)
    logdet = logabsdetjacobian(M, x)
    return evaluate(M, x), logdet
end

function _evaluate_logabsdet(C::ComposedMap, x)
    _check_composed_input(C, x)
    value = zero(eltype(x))
    for layer in C.layers
        x, logdet = _evaluate_logabsdet(layer, x)
        value += logdet
    end
    return x, value
end

logabsdetjacobian(C::ComposedMap, x::AbstractVector{<:Real}) = last(_evaluate_logabsdet(C, x))

function logabsdetjacobian(C::ComposedMap, X::AbstractMatrix{<:Real})
    _check_composed_input(C, X)
    return [logabsdetjacobian(C, x) for x in eachrow(X)]
end

"""
    inverse_jacobian(C::ComposedMap, x)

Determinant of the inverse map at `x`, or each row of a matrix.
"""
inverse_jacobian(C::ComposedMap, x::AbstractVector{<:Real}) = inv(jacobian(C, inverse(C, x)))
inverse_jacobian(C::ComposedMap, X::AbstractMatrix{<:Real}) = (_check_composed_input(C, X); [inverse_jacobian(C, x) for x in eachrow(X)])

# Accumulate inverse determinants as we walk backwards through a chain. A
# support-limited layer may return `nothing` for points outside its image.
function _inverse_logabsdet(M, x)
    z = inverse(M, x)
    return z, -logabsdetjacobian(M, z)
end

function _inverse_logabsdet(C::ComposedMap, x)
    _check_composed_input(C, x)
    logdet = zero(eltype(x))
    for layer in Iterators.reverse(C.layers)
        x, contribution = _inverse_logabsdet(layer, x)
        x === nothing && return nothing, -Inf
        logdet += contribution
    end
    return x, logdet
end

"""
    logpdf(C::ComposedMap, x)
    pdf(C::ComposedMap, x)
    pullback(C::ComposedMap, x)

Evaluate the target-density approximation of the **whole** composition.
For `:reference`, this is `ρ(C(x)) * abs(det(DC(x)))`; for `:target`, it is
`ρ(inv(C)(x)) * abs(det(Dinv(C)(x)))`. `logpdf` accumulates determinants in
log space. `pullback` is an alias for `pdf`, following the package convention.
Matrix input returns one value per row.
"""
function logpdf(C::ComposedMap, x::AbstractVector{<:Real})
    if C.forwarddirection === :reference
        z, logdet = _evaluate_logabsdet(C, x)
        return logpdf(C.reference, collect(z)) + logdet
    end
    z, logdet = _inverse_logabsdet(C, x)
    z === nothing && return -Inf
    return logpdf(C.reference, collect(z)) + logdet
end
logpdf(C::ComposedMap, X::AbstractMatrix{<:Real}) = (_check_composed_input(C, X); [logpdf(C, x) for x in eachrow(X)])
pdf(C::ComposedMap, x::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) = exp.(logpdf(C, x))
pullback(C::ComposedMap, x::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) = pdf(C, x)

"""
    pushforward(C::ComposedMap, target::AbstractMapDensity, z)

Evaluate `target` represented in reference coordinates (the package convention
for `pushforward`). For `:target` maps, return `target(C(z)) * abs(det(DC(z)))`.
For `:reference` maps, use the inverse composition and its determinant.
"""
function pushforward(C::ComposedMap, target::AbstractMapDensity, z::AbstractVector{<:Real})
    if C.forwarddirection === :target
        x, logdet = _evaluate_logabsdet(C, z)
        return exp(logpdf(target, collect(x)) + logdet)
    end
    x = inverse(C, z)
    return exp(logpdf(target, collect(x)) - logabsdetjacobian(C, x))
end

function pushforward(C::ComposedMap, target::AbstractMapDensity, Z::AbstractMatrix{<:Real})
    _check_composed_input(C, Z)
    return [pushforward(C, target, z) for z in eachrow(Z)]
end

(C::ComposedMap)(x::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) = evaluate(C, x)

function Base.show(io::IO, C::ComposedMap)
    print(io, "ComposedMap(")
    join(io, C.layers, ", ")
    print(io, "; forwarddirection=:", C.forwarddirection, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", C::ComposedMap)
    println(io, "ComposedMap with $(numberdimensions(C)) dimensions and $(length(C.layers)) layers (first applied first):")
    for (i, layer) in enumerate(C.layers)
        println(io, "  ", i, ": ", layer)
    end
    print(io, "  forwarddirection: :", C.forwarddirection)
    return nothing
end
