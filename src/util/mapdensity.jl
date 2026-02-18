"""
    MapTargetDensity

Wrapper for target density functions used in transport map optimization.
Stores the log-density function and its gradient, with support for automatic
differentiation backends via DifferentiationInterface.jl.

# Fields
- `logdensity<:Function`: Function computing log-density `log π(x)`
- `ad_backend<:Union{Nothing,ADTypes.AbstractADType}`: AD backend or `nothing` for analytical
- `grad_logdensity<:Function`: Function computing gradient `∇ log π(x)`
- `prepared_gradient`: Optional prepared gradient for performance (can be `nothing`)
- `isvectorized::Bool`: Whether the log-density (and gradient, if analytical) accepts matrix input

# Constructors
- `MapTargetDensity(logdensity, grad_logdensity; isvectorized=false)`: Provide both log-density and analytical gradient.
- `MapTargetDensity(logdensity, backend::ADTypes.AbstractADType, d::Int; isvectorized=false)`: Use AD backend with prepared gradient.
- `MapTargetDensity(logdensity, backend::ADTypes.AbstractADType; isvectorized=false)`: Use AD backend without preparation.
- `MapTargetDensity(logdensity; isvectorized=false)`: Use ForwardDiff.

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
density = MapTargetDensity(logπ_vectorized; isvectorized=true)
```
"""
struct MapTargetDensity{F<:Function,B<:Union{Nothing,ADTypes.AbstractADType},G<:Function,P<:Union{Nothing,DifferentiationInterface.GradientPrep}} <: AbstractMapDensity
    logdensity::F
    ad_backend::B
    grad_logdensity::G
    prepared_gradient::P
    isvectorized::Bool

    # Analytical gradient
    function MapTargetDensity(logdensity::F, grad_logdensity::G; isvectorized::Bool=false) where {F<:Function,G<:Function}
        return new{F,Nothing,G,Nothing}(logdensity, nothing, grad_logdensity, nothing, isvectorized)
    end

    # AD backend with prepared gradient
    function MapTargetDensity(logdensity::F, backend::B, d::Int; isvectorized::Bool=false) where {F<:Function,B<:ADTypes.AbstractADType}
        # Prepare gradient once for this input size
        prep = DifferentiationInterface.prepare_gradient(logdensity, backend, zeros(d))

        grad_logdensity = function (x)
            return DifferentiationInterface.gradient(logdensity, prep, backend, x)
        end
        return new{F,B,typeof(grad_logdensity),typeof(prep)}(logdensity, backend, grad_logdensity, prep, isvectorized)
    end

    # AD backend without preparation
    function MapTargetDensity(logdensity::F, backend::B; isvectorized::Bool=false) where {F<:Function,B<:ADTypes.AbstractADType}
        grad_logdensity = function (x)
            return DifferentiationInterface.gradient(logdensity, backend, x)
        end
        return new{F,B,typeof(grad_logdensity),Nothing}(logdensity, backend, grad_logdensity, nothing, isvectorized)
    end

    # Default: ForwardDiff without preparation
    function MapTargetDensity(logdensity::F; isvectorized::Bool=false) where {F<:Function}
        backend = AutoForwardDiff()
        grad_logdensity = function (x)
            return DifferentiationInterface.gradient(logdensity, backend, x)
        end
        return new{F,typeof(backend),typeof(grad_logdensity),Nothing}(logdensity, backend, grad_logdensity, nothing, isvectorized)
    end
end

"""
    MapReferenceDensity

Wrapper for reference density (typically standard Gaussian) used in transport maps.
The reference density defines the space from which samples are drawn and mapped
to the target distribution.

# Fields
- `logdensity<:Function`: Function computing log-density `log ρ(z)`
- `ad_backend<:ADTypes.AbstractADType`: AD backend for gradient computation
- `grad_logdensity<:Function`: Function computing gradient `∇ log ρ(z)`
- `densitytype::Distributions.UnivariateDistribution`: Univariate density type (e.g., `Normal()`)
- `prepared_gradient`: Prepared gradient for performance (can be `nothing`)

# Constructors
- `MapReferenceDensity()`: Use standard normal with ForwardDiff (default, not prepared).
- `MapReferenceDensity(densitytype)`: Specify distribution with ForwardDiff (not prepared).
- `MapReferenceDensity(densitytype, backend)`: Specify distribution and AD backend (not prepared).
- `MapReferenceDensity(densitytype, backend, d)`: Specify distribution, AD backend, and dimension (prepared, recommended for performance).
"""
struct MapReferenceDensity{F<:Function,B<:ADTypes.AbstractADType,G<:Function,P<:Union{Nothing,DifferentiationInterface.GradientPrep}} <: AbstractMapDensity
    logdensity::F
    ad_backend::B
    grad_logdensity::G
    densitytype::Distributions.UnivariateDistribution
    prepared_gradient::P
    isvectorized::Bool

    # Constructor with dimension specified for preparation (recommended for performance)
    function MapReferenceDensity(
        densitytype::Distributions.UnivariateDistribution,
        backend::ADTypes.AbstractADType,
        d::Int
    )
        density = x -> sum(map(Base.Fix1(logpdf, densitytype), x))

        # Prepare gradient for dimension d
        x_example = zeros(d)
        prep = DifferentiationInterface.prepare_gradient(density, backend, x_example)

        grad_density = function (x)
            return DifferentiationInterface.gradient(density, prep, backend, x)
        end
        return new{typeof(density),typeof(backend),typeof(grad_density),typeof(prep)}(
            density, backend, grad_density, densitytype, prep, false
        )
    end

    # base constructor for reference density, without preparation (convenience)
    function MapReferenceDensity(
        densitytype::Distributions.UnivariateDistribution=Normal(),
        backend::ADTypes.AbstractADType=AutoForwardDiff()
    )
        density = x -> sum(map(Base.Fix1(logpdf, densitytype), x))
        grad_density = function (x)
            return DifferentiationInterface.gradient(density, backend, x)
        end
        return new{typeof(density),typeof(backend),typeof(grad_density),Nothing}(
            density, backend, grad_density, densitytype, nothing, false
        )
    end

    function MapReferenceDensity(densitytype::Distributions.Uniform)
        error("Not implemented yet")
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
logpdf(density::AbstractMapDensity, x::Vector{<:Real}) = density.logdensity(x)

logpdf(density::AbstractMapDensity, x::Real) = logpdf(density, [x])

function logpdf(density::AbstractMapDensity, X::Matrix{<:Real})

    if density.isvectorized
        return density.logdensity(X)

    else
        n = size(X, 1)
        logdensities = zeros(Float64, n)

        Threads.@threads for i in 1:n
            logdensities[i] = density.logdensity(view(X, i, :))
        end
        return logdensities
    end
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
grad_logpdf(density::AbstractMapDensity, x::Vector{<:Real}) = density.grad_logdensity(x)

function grad_logpdf(density::AbstractMapDensity, X::Matrix{<:Real})

    if density.isvectorized && isnothing(density.ad_backend)
        return density.grad_logdensity(X)
    else
        n, d = size(X)
        log_gradients = zeros(Float64, n, d)

        threaded = !(density.ad_backend isa ADTypes.AutoMooncake)

        if threaded
            Threads.@threads for i in 1:n
                log_gradients[i, :] = density.grad_logdensity(X[i, :])
            end
        else
            for i in 1:n
                log_gradients[i, :] = density.grad_logdensity(X[i, :])
            end
        end

        return log_gradients
    end
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
pdf(density::AbstractMapDensity, x::Vector{<:Real}) = exp(density.logdensity(x))

pdf(density::AbstractMapDensity, x::Real) = pdf(density, [x])

function pdf(density::AbstractMapDensity, X::Matrix{<:Real})

    if density.isvectorized
        return exp.(density.logdensity(X))
    else
        n = size(X, 1)
        densities = zeros(Float64, n)

        Threads.@threads for i in 1:n
            densities[i] = exp(density.logdensity(view(X, i, :)))
        end
        return densities
    end
end

function Base.show(io::IO, target::MapTargetDensity)
    backend_str = target.ad_backend === nothing ? "analytical" : string(target.ad_backend)
    print(io, "MapTargetDensity(backend=$(backend_str))")
end

function Base.show(io::IO, ref::MapReferenceDensity)
    print(io, "MapReferenceDensity(density=$(ref.densitytype), backend=$(ref.ad_backend))")
end
