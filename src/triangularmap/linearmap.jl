"""
    LinearMap <: AbstractLinearMap

A linear transformation map that standardizes data using mean and standard deviation.

The forward and inverse transformations are

```math
z_i = \\frac{x_i-\\mu_i}{\\sigma_i},
\\qquad
x_i = \\sigma_i z_i + \\mu_i.
```

# Fields
- `μ::Vector{Float64}`: Mean vector for each dimension
- `σ::Vector{Float64}`: Standard deviation vector for each dimension

# Constructors
- `LinearMap(samples::Matrix{Float64})`: Compute empirical mean and standard deviation from samples.
- `LinearMap(μ::Vector{Float64}, σ::Vector{Float64})`: Construct linear map with explicit mean and standard deviation.
"""
struct LinearMap <: AbstractLinearMap
    μ::Vector{Float64}
    σ::Vector{Float64}

    # Constructor that computes mean and std from samples
    function LinearMap(samples::Matrix{Float64})
        μ = mean(samples, dims = 1)[:]
        σ = std(samples, dims = 1)[:]
        return new(μ, σ)
    end

    # Identity map
    function LinearMap()
        return new(zeros(Float64, 0), ones(Float64, 0))
    end

    # Identity map with specified dimension
    function LinearMap(dim::Int)
        return new(zeros(Float64, dim), ones(Float64, dim))
    end

    function LinearMap(μ::Vector{Float64}, σ::Vector{Float64})
        return new(μ, σ)
    end
end

"""
    setparameters!(map::LinearMap, μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real})

Set the mean and standard deviation parameters of the linear map.
"""
function setparameters!(map::LinearMap, μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real})
    map.μ .= μ
    map.σ .= σ
    return nothing
end

"""
    evaluate(L::LinearMap, x::AbstractVector{<:Real})

Standardize the input componentwise:
``L(x)_i=(x_i-\\mu_i)/\\sigma_i``.
"""
function evaluate(L::LinearMap, x::AbstractVector{<:Real})
    if numberdimensions(L) == 0
        return x  # Identity map if no dimensions are defined
    else
        @assert length(x) == length(L.μ) "Input vector must have the same length as dimensions in the map"
        return (x .- L.μ) ./ L.σ
    end
end

"""
    evaluate(L::LinearMap, X::AbstractMatrix{<:Real})

Apply the linear transformation to multiple points (row-wise).
"""
function evaluate(L::LinearMap, X::AbstractMatrix{<:Real})
    if numberdimensions(L) == 0
        return X  # Identity map if no dimensions are defined
    else
        @assert size(X, 2) == length(L.μ) "Input data must have the same number of columns as dimensions in the map"
        return (X .- L.μ') ./ L.σ'
    end
end

"""
    inverse(L::LinearMap, y::AbstractVector{<:Real})

Recover the original scale with
``L^{-1}(y)_i=\\sigma_i y_i+\\mu_i``.
"""
function inverse(L::LinearMap, y::AbstractVector{<:Real})
    if numberdimensions(L) == 0
        return y # Identity map if no dimensions are defined
    else
        @assert length(y) == length(L.μ) "Input vector must have the same length as dimensions in the map"
        return (y .* L.σ) .+ L.μ
    end
end

"""
    inverse(L::LinearMap, Y::AbstractMatrix{<:Real})

Invert the transformation for multiple points (row-wise).
"""
function inverse(L::LinearMap, Y::AbstractMatrix{<:Real})
    if numberdimensions(L) == 0
        return Y  # Identity map if no dimensions are defined
    else
        @assert size(Y, 2) == length(L.μ) "Input data must have the same number of columns as dimensions in the map"
        return (Y .* L.σ') .+ L.μ'
    end
end

"""
    jacobian(L::LinearMap)

Return the scale determinant ``\\prod_{i=1}^d\\sigma_i``.
"""
function jacobian(L::LinearMap)
    return prod(L.σ)
end

"""
    numberdimensions(L::LinearMap)

Return the number of dimensions of the linear map.
"""
numberdimensions(L::LinearMap) = length(L.μ)

function Base.show(io::IO, L::LinearMap)
    print(io, "LinearMap($(numberdimensions(L))-dimensional, ")
    print(io, "μ: ", L.μ, ", ")
    print(io, "σ: ", L.σ, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", L::LinearMap)
    println(io, "LinearMap with $(numberdimensions(L)) dimensions")
    println(io, "  μ: ", L.μ)
    println(io, "  σ: ", L.σ)
    return nothing
end
