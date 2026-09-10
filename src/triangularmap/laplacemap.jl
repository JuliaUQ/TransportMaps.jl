"""
    LaplaceMap <: AbstractLinearMap

A linear transport map based on the Laplace approximation of a target density.

The Laplace approximation assumes the target density is approximately Gaussian around its mode.
Let ``\\mu`` denote the mode and let ``LL^\\mathsf{T}=\\Sigma`` be the Cholesky
factorization of the covariance approximation. The forward and inverse maps are

```math
z = L^{-1}(x-\\mu),
\\qquad
x = Lz+\\mu.
```

# Fields
- `mode::Vector{Float64}`: Mode or mean vector of the approximation
- `chol::Matrix{Float64}`: Lower Cholesky factor ``L`` satisfying
  ``\\Sigma=LL^\\mathsf{T}``

# Constructors

- `LaplaceMap(samples::Matrix{Float64})`: Constructs a LaplaceMap from sample data by
computing the empirical mean and covariance.
- `LaplaceMap(density::MapTargetDensity, x0::Vector{Float64}; optimizer::Optim.AbstractOptimizer = LBFGS(), options::Optim.Options = Optim.Options())`: Compute a Laplace approximation of a target density by finding the mode via optimization and computing the Hessian at the mode. If `hessian_backend` is `nothing`, uses the same backend as the density.
"""
struct LaplaceMap <: AbstractLinearMap
    mode::Vector{Float64}  # Mode / mean vector
    chol::Matrix{Float64}  # Cholesky decomposition of the covariance matrix

    # Construct a LaplaceMap from sample data
    function LaplaceMap(samples::Matrix{Float64})
        mode = mean(samples, dims = 1)[:]
        Σ = cov(samples, corrected = true)
        # Cholesky decomposition (Σ = L * L')
        chol = cholesky(Σ).L
        return new(mode, chol)
    end

    # Compute a Laplace approximation given a target density and initial guess x0
    function LaplaceMap(
            density::MapTargetDensity,
            x0::Vector{Float64};
            optimizer::Optim.AbstractOptimizer = LBFGS(),
            options::Optim.Options = Optim.Options()
        )

        # objective: f(x) = -log(π(x))
        function obj(x)
            return -logpdf(density, x)
        end

        # gradient: f'(x) = - ∇(log(π(x))) (chain rule)
        function grad!(storage, x)
            storage .= -grad_logpdf(density, x)
            return storage
        end

        # Optimize to find the mode
        res = optimize(obj, grad!, x0, optimizer, options)

        if Optim.converged(res)
            mode = Optim.minimizer(res)
        else
            error("LaplaceMap optimization did not converge.")
        end

        # Compute Hessian at mode
        if isnothing(density.ad_backend)
            hessian_backend = AutoFiniteDiff()
        else
            hessian_backend = density.ad_backend
        end

        H = DifferentiationInterface.hessian(obj, hessian_backend, mode)

        # Make matrix Hermitian to avoid numerical issues
        Σ = Hermitian(inv(H))
        chol = cholesky(Σ).L

        return new(mode, chol)
    end
end

"""
    evaluate(L::LaplaceMap, x::AbstractVector{<:Real})

Apply the whitening transformation ``L^{-1}(x-\\mu)``.
"""
function evaluate(L::LaplaceMap, x::AbstractVector{<:Real})
    @assert length(x) == length(L.mode) "Input vector must have the same length as dimensions in the map"
    return L.chol \ (x .- L.mode)
end

"""
    evaluate(L::LaplaceMap, X::AbstractMatrix{<:Real})

Apply the Laplace map transformation to multiple points (row-wise).
"""
function evaluate(L::LaplaceMap, X::AbstractMatrix{<:Real})
    @assert size(X, 2) == length(L.mode) "Input data must have the same number of columns as dimensions in the map"
    return (X .- L.mode') * inv(L.chol)'
end

"""
    inverse(L::LaplaceMap, y::AbstractVector{<:Real})

Invert the Laplace map, returning ``Ly+\\mu``.
"""
function inverse(L::LaplaceMap, y::AbstractVector{<:Real})
    @assert length(y) == length(L.mode) "Input vector must have the same length as dimensions in the map"
    return L.chol * y .+ L.mode
end

"""
    inverse(L::LaplaceMap, Y::AbstractMatrix{<:Real})

Invert the transformation for multiple points (row-wise).
"""
function inverse(L::LaplaceMap, Y::AbstractMatrix{<:Real})
    @assert size(Y, 2) == length(L.mode) "Input data must have the same number of columns as dimensions in the map"
    return Y * L.chol' .+ L.mode'
end

"""
    jacobian(L::LaplaceMap)

Return the scale determinant ``|\\det L|``.
"""
function jacobian(L::LaplaceMap)
    return abs(det(L.chol))
end

"""
    numberdimensions(L::LaplaceMap)

Return the number of dimensions of the Laplace map.
"""
numberdimensions(L::LaplaceMap) = length(L.mode)

"""
    cov(L::LaplaceMap)

Return the covariance matrix ``\\Sigma=LL^\\mathsf{T}``.
"""
cov(L::LaplaceMap) = L.chol * L.chol'

"""
    mean(L::LaplaceMap)

Return the mean (mode) of the Laplace approximation.
"""
mean(L::LaplaceMap) = L.mode

"""
    mode(L::LaplaceMap)

Return the mode of the Laplace approximation.
"""
mode(L::LaplaceMap) = L.mode

"""
    MvNormal(L::LaplaceMap)

Construct a multivariate normal distribution from the Laplace approximation.
"""
MvNormal(L::LaplaceMap) = Distributions.MvNormal(mean(L), cov(L))

# Make LaplaceMap callable: L(x) instead of evaluate(L, x)
Base.@propagate_inbounds (L::LaplaceMap)(x::AbstractVector{<:Real}) = evaluate(L, x)
Base.@propagate_inbounds (L::LaplaceMap)(X::AbstractMatrix{<:Real}) = evaluate(L, X)

function Base.show(io::IO, L::LaplaceMap)
    print(io, "LaplaceMap($(numberdimensions(L))-dimensional")
    print(io, " mode: ", mode(L), ", ")
    print(io, " Σ: ", cov(L), ")")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", L::LaplaceMap)
    println(io, "LaplaceMap with $(numberdimensions(L)) dimensions")
    println(io, "  mode: ", mode(L))
    println(io, "  Σ: ", cov(L))
    return nothing
end
