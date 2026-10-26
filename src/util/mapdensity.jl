"""
    MapTargetDensity

Wrapper for target density functions used in transport map optimization.
Stores the log-density function and its gradient, with support for automatic
differentiation backends via DifferentiationInterface.jl.

# Fields
- `logdensity<:Function`: Function computing ``\\log\\pi(x)``
- `ad_backend<:Union{Nothing,ADTypes.AbstractADType}`: AD backend or `nothing` for analytical
- `grad_logdensity<:Function`: Function computing ``\\nabla_x\\log\\pi(x)``
- `prepared_gradient`: Optional prepared gradient for performance (can be `nothing`)
- `isvectorized::Bool`: Whether the log-density (and gradient, if analytical) accepts matrix input
- `threaded::Bool`: Whether to use multithreading for row-wise density and gradient evaluations on matrices (default: `true`)

# Constructors
- `MapTargetDensity(logdensity, grad_logdensity; isvectorized=false, threaded=true)`: Provide both log-density and analytical gradient.
- `MapTargetDensity(logdensity, backend::ADTypes.AbstractADType, d::Int; isvectorized=false, threaded=true)`: Use AD backend with prepared gradient.
- `MapTargetDensity(logdensity, backend::ADTypes.AbstractADType; isvectorized=false, threaded=true)`: Use AD backend without preparation.
- `MapTargetDensity(logdensity; isvectorized=false, threaded=true)`: Use ForwardDiff.

# Examples
```julia
# Use ForwardDiff (default, not prepared)
density = MapTargetDensity(logπ)

# Use ForwardDiff with prepared gradient (faster for repeated evaluations)
density = MapTargetDensity(logπ, AutoForwardDiff(), d)

# Use Mooncake with preparation
density = MapTargetDensity(logπ, AutoMooncake(), d)

# Use FiniteDiff
density = MapTargetDensity(logπ, AutoFiniteDiff(), d)

# Use analytical gradient
density = MapTargetDensity(logπ, grad_logπ)

# Use vectorized log-density (accepts matrix input where rows are samples)
density = MapTargetDensity(logπ_vectorized; isvectorized = true)
```
"""
struct MapTargetDensity{F <: Function, B <: Union{Nothing, ADTypes.AbstractADType}, G <: Function, P <: Union{Nothing, DifferentiationInterface.GradientPrep}} <: AbstractMapDensity
    logdensity::F
    ad_backend::B
    grad_logdensity::G
    prepared_gradient::P
    isvectorized::Bool
    threaded::Bool

    # Analytical gradient
    function MapTargetDensity(logdensity::F, grad_logdensity::G; isvectorized::Bool = false, threaded::Bool = true) where {F <: Function, G <: Function}
        return new{F, Nothing, G, Nothing}(logdensity, nothing, grad_logdensity, nothing, isvectorized, threaded)
    end

    # AD backend with prepared gradient
    function MapTargetDensity(logdensity::F, backend::B, d::Int; isvectorized::Bool = false, threaded::Bool = true) where {F <: Function, B <: ADTypes.AbstractADType}
        # Prepare gradient once for this input size
        prep = DifferentiationInterface.prepare_gradient(logdensity, backend, zeros(d))

        grad_logdensity = function (x)
            return DifferentiationInterface.gradient(logdensity, prep, backend, x)
        end

        if isa(backend, AutoMooncake)
            threaded = false
        end

        return new{F, B, typeof(grad_logdensity), typeof(prep)}(logdensity, backend, grad_logdensity, prep, isvectorized, threaded)
    end

    # AD backend without preparation
    function MapTargetDensity(logdensity::F, backend::B; isvectorized::Bool = false, threaded::Bool = true) where {F <: Function, B <: ADTypes.AbstractADType}
        grad_logdensity = function (x)
            return DifferentiationInterface.gradient(logdensity, backend, x)
        end

        if isa(backend, AutoMooncake)
            threaded = false
        end

        return new{F, B, typeof(grad_logdensity), Nothing}(logdensity, backend, grad_logdensity, nothing, isvectorized, threaded)
    end

    # Default: ForwardDiff without preparation
    function MapTargetDensity(logdensity::F; isvectorized::Bool = false, threaded::Bool = true) where {F <: Function}
        backend = AutoForwardDiff()
        grad_logdensity = function (x)
            return DifferentiationInterface.gradient(logdensity, backend, x)
        end
        return new{F, typeof(backend), typeof(grad_logdensity), Nothing}(logdensity, backend, grad_logdensity, nothing, isvectorized, threaded)
    end
end

"""
    MapReferenceDensity

Wrapper for reference density (typically standard Gaussian) used in transport maps.
The reference density defines the space from which samples are drawn and mapped
to the target distribution.

# Fields
- `logdensity<:Function`: Function computing ``\\log\\rho(z)``
- `grad_logdensity<:Function`: Function computing ``\\nabla_z\\log\\rho(z)`` via
  `gradlogpdf` when available, otherwise via ForwardDiff
- `densitytype::Distributions.UnivariateDistribution`: Univariate density type (e.g., `Normal()`)

# Constructors
- `MapReferenceDensity()`: Use standard normal as reference density.
- `MapReferenceDensity(densitytype)`: Specify univariate distribution; uses analytical `gradlogpdf` when available, otherwise falls back to ForwardDiff.
"""
struct MapReferenceDensity{F <: Function, G <: Function} <: AbstractMapDensity
    logdensity::F
    grad_logdensity::G
    densitytype::Distributions.UnivariateDistribution

    function MapReferenceDensity(
            densitytype::Distributions.UnivariateDistribution = Normal()
        )
        density = x -> sum(value -> logpdf(densitytype, value), x; init = zero(eltype(x)))

        # Use gradlogpdf if available, otherwise fall back to ForwardDiff
        grad_density = if hasmethod(gradlogpdf, Tuple{typeof(densitytype), Float64})
            x -> gradlogpdf.(Ref(densitytype), x)
        else
            backend = AutoForwardDiff()
            x -> DifferentiationInterface.gradient(density, backend, x)
        end

        return new{typeof(density), typeof(grad_density)}(
            density, grad_density, densitytype
        )
    end
end

"""
    logpdf(density::AbstractMapDensity, x)

Evaluate the log-density at point(s) `x`.

# Arguments
- `density::AbstractMapDensity`: Target or reference density
- `x`: Point (vector) or multiple points (matrix, rows are samples) at which to evaluate

# Returns
- Scalar log-density value for vector input, or vector of log-densities for matrix input
"""
logpdf(density::AbstractMapDensity, x::AbstractVector{<:Real}) = density.logdensity(x)

logpdf(density::AbstractMapDensity, x::Real) = logpdf(density, [x])
logpdf(density::MapReferenceDensity, x::Real) = logpdf(density.densitytype, x)

function logpdf(density::MapTargetDensity, X::AbstractMatrix{<:Real})
    if density.isvectorized
        return density.logdensity(X)
    else
        return _batch_logpdf(density.logdensity, X, density.threaded)
    end
end

function logpdf(density::MapReferenceDensity, X::AbstractMatrix{<:Real})
    return _batch_logpdf(density.logdensity, X, true)
end

function _batch_logpdf(logdensity, X, threaded)
    n = size(X, 1)
    logdensities = Vector{Float64}(undef, n)
    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            logdensities[i] = logdensity(view(X, i, :))
        end
    else
        for i in 1:n
            logdensities[i] = logdensity(view(X, i, :))
        end
    end
    return logdensities
end

"""
    grad_logpdf(density::AbstractMapDensity, x)

Evaluate the gradient of log-density at point(s) `x`.

# Arguments
- `density::AbstractMapDensity`: Target or reference density
- `x`: Point (vector) or multiple points (matrix, rows are samples) at which to evaluate

# Returns
- Gradient vector for vector input, or matrix of gradients (one per row) for matrix input
"""
grad_logpdf(density::AbstractMapDensity, x::AbstractVector{<:Real}) = density.grad_logdensity(x)

function grad_logpdf(density::MapTargetDensity, X::AbstractMatrix{<:Real})
    if density.isvectorized && isnothing(density.ad_backend)
        return density.grad_logdensity(X)
    elseif !isnothing(density.prepared_gradient)
        return _batch_density_gradient(density, X, density.threaded)
    else
        return _batch_density_gradient(density.grad_logdensity, X, density.threaded)
    end
end

function grad_logpdf(density::MapReferenceDensity, X::AbstractMatrix{<:Real})
    return _reference_density_gradient(density.densitytype, density.grad_logdensity, X)
end

function _reference_density_gradient(distribution, gradient, X)
    # These references are products of univariate distributions. Their analytic
    # scores can be broadcast directly over the matrix without copying rows or
    # allocating an intermediate gradient vector for every sample.
    if hasmethod(gradlogpdf, Tuple{typeof(distribution), Float64})
        output = Matrix{Float64}(undef, size(X))
        output .= gradlogpdf.(Ref(distribution), X)
        return output
    end
    return _batch_density_gradient(gradient, X, true)
end

function _batch_density_gradient(gradient, X, threaded)
    n, d = size(X)
    log_gradients = Matrix{Float64}(undef, n, d)
    n == 0 && return log_gradients

    if threaded && Threads.nthreads() > 1
        n_chunks = min(n, Threads.nthreads())
        Threads.@threads for chunk in 1:n_chunks
            first_row = fld((chunk - 1) * n, n_chunks) + 1
            last_row = fld(chunk * n, n_chunks)
            _density_gradient_rows!(log_gradients, gradient, X, first_row:last_row)
        end
    else
        _density_gradient_rows!(log_gradients, gradient, X, 1:n)
    end
    return log_gradients
end

function _density_gradient_rows!(output, gradient, X, rows)
    # A task owns this buffer, so task migration cannot create a race. Keeping a
    # Vector also preserves callbacks/prepared AD backends requiring Vector input.
    point = Vector{eltype(X)}(undef, size(X, 2))
    for i in rows
        copyto!(point, view(X, i, :))
        values = gradient(point)
        length(values) == length(point) || throw(DimensionMismatch("Density gradient must match the point dimension"))
        copyto!(view(output, i, :), values)
    end
    return output
end

function _density_gradient_rows!(
        output, density::MapTargetDensity{F, B, G, P}, X, rows,
    ) where {F, B, G, P <: DifferentiationInterface.GradientPrep}
    point = Vector{eltype(X)}(undef, size(X, 2))
    score = Vector{Float64}(undef, length(point))
    copyto!(point, view(X, first(rows), :))
    # DifferentiationInterface preparations contain mutable caches. Give each
    # parallel task its own preparation; serial calls reuse the existing one.
    prep = if density.threaded && Threads.nthreads() > 1
        DifferentiationInterface.prepare_gradient(density.logdensity, density.ad_backend, point)
    else
        density.prepared_gradient
    end
    for i in rows
        copyto!(point, view(X, i, :))
        DifferentiationInterface.gradient!(density.logdensity, score, prep, density.ad_backend, point)
        copyto!(view(output, i, :), score)
    end
    return output
end

"""
    pdf(density::AbstractMapDensity, x)

Evaluate the probability density at point(s) `x`.

# Arguments
- `density::AbstractMapDensity`: Target or reference density
- `x`: Point (vector) or multiple points (matrix, rows are samples) at which to evaluate

# Returns
- Scalar density value for vector input, or vector of densities for matrix input

# Note
This computes `exp(logpdf(density, x))`. For numerical stability, prefer using `logpdf` when possible.
"""
pdf(density::AbstractMapDensity, x::AbstractVector{<:Real}) = exp(logpdf(density, x))

pdf(density::AbstractMapDensity, x::Real) = exp(logpdf(density, x))

pdf(density::AbstractMapDensity, X::AbstractMatrix{<:Real}) = exp.(logpdf(density, X))

function Base.show(io::IO, target::MapTargetDensity)
    backend_str = target.ad_backend === nothing ? "analytical" : string(target.ad_backend)
    print(io, "MapTargetDensity(backend=$(backend_str))")
    return nothing
end

function Base.show(io::IO, ref::MapReferenceDensity)
    print(io, "MapReferenceDensity(density=$(ref.densitytype))")
    return nothing
end
