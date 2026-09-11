# Conditional densities for triangular maps making use of the Knothe-Rosenblatt transform

"""
    conditional_density(M::PolynomialMap, x_range::Real, x_given::AbstractVector{<:Real})

Compute the conditional density
``\\pi(x_k\\mid x_1,\\ldots,x_{k-1})`` at `x_range` given `x_given`.

With ``z_{1:k}=M^{-1}(x_{1:k})``, this is evaluated as

```math
\\pi(x_k\\mid x_{1:k-1})
= \\rho(z_k)
  \\left|\\frac{\\partial M^k(z_{1:k})}{\\partial z_k}\\right|^{-1}.
```
"""
function conditional_density(M::PolynomialMap, x_range::Real, x_given::AbstractVector{<:Real})
    k = length(x_given) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    # invert the first k components to get z_range
    z_range = inverse(M, [x_given..., x_range], k)

    # conditional density: π(xₖ | x₁, ..., xₖ₋₁) = ρ(zₖ) * |∂Tₖ/∂zₖ|^{-1}
    cond_density = pdf(M.reference, z_range[k]) * abs(1 / partial_derivative_zk(M.components[k], z_range))

    return cond_density
end

# For convenience when x_given is a single value
conditional_density(M::PolynomialMap, x_range::Real, x_given::Real) = conditional_density(M, x_range, [x_given])

"""
    conditional_density(M::PolynomialMap, x_range::AbstractVector{<:Real}, x_given::AbstractVector{<:Real})

Compute conditional densities at multiple x_range values using multithreading.
"""
function conditional_density(M::PolynomialMap, x_range::AbstractVector{<:Real}, x_given::AbstractVector{<:Real})
    k = length(x_given) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    n_points = length(x_range)
    cond_densities = Vector{Float64}(undef, n_points)

    # Use multithreading to compute conditional densities for each point
    Threads.@threads for i in 1:n_points
        cond_densities[i] = conditional_density(M, x_range[i], x_given)
    end

    return cond_densities
end

conditional_density(M::PolynomialMap, x_range::AbstractVector{<:Real}, x_given::Real) = conditional_density(M, x_range, [x_given])

"""
    conditional_sample(M::PolynomialMap, x_given::AbstractVector{<:Real}, z_range::Real)

Generate a sample from ``\\pi(x_k\\mid x_1,\\ldots,x_{k-1})`` by pushing
``z_k\\sim\\rho`` through the ``k``th map component.
"""
function conditional_sample(M::PolynomialMap, x_given::AbstractVector{<:Real}, z_range::Real)

    k = length(x_given) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    # invert the first k components to get z_given
    z_given = inverse(M, x_given, k - 1)
    # Push through the k-th component of the map
    return evaluate(M.components[k], [z_given..., z_range])
end

# For convenience when x_given is a single value
conditional_sample(M::PolynomialMap, x_given::Real, z_range::Real) = conditional_sample(M, [x_given], z_range)

"""
    conditional_sample(M::PolynomialMap, x_given::AbstractVector{<:Real}, z_range::AbstractVector{<:Real})

Generate multiple samples from ``\\pi(x_k\\mid x_1,\\ldots,x_{k-1})``.
"""
function conditional_sample(M::PolynomialMap, x_given::AbstractVector{<:Real}, z_range::AbstractVector{<:Real})
    k = length(x_given) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    n_points = length(z_range)
    samples = Vector{Float64}(undef, n_points)
    # Use multithreading to compute samples for each point
    Threads.@threads for i in 1:n_points
        samples[i] = conditional_sample(M, x_given, z_range[i])
    end
    return samples
end

conditional_sample(M::PolynomialMap, x_given::Real, z_range::AbstractVector{<:Real}) = conditional_sample(M, [x_given], z_range)

"""
    multivariate_conditional_density(M::PolynomialMap, x::AbstractVector{<:Real})

Compute the multivariate conditional density using

```math
\\pi(x_{j:k}\\mid x_{1:j-1})
= \\prod_{i=j}^k \\pi(x_i\\mid x_{1:i-1}).
```
"""
function multivariate_conditional_density(M::PolynomialMap, x::AbstractVector{<:Real})
    d = length(x)
    @assert d <= numberdimensions(M) "Length of x cannot exceed the dimension of the map"

    if d == 1
        # For single dimension, this is just the marginal density of x₁
        z = inverse(M, x, 1)
        return pdf(M.reference, z[1]) * abs(1 / partial_derivative_zk(M.components[1], z))
    end

    # Start with the first variable (marginal density)
    z₁ = inverse(M, x[1:1], 1)
    density = pdf(M.reference, z₁[1]) * abs(1 / partial_derivative_zk(M.components[1], z₁))

    # Multiply by conditional densities for subsequent variables
    for k in 2:d
        x_given_k = x[1:(k - 1)]
        x_range_k = x[k]
        density *= conditional_density(M, x_range_k, x_given_k)
    end

    return density
end

"""
    multivariate_conditional_density(M::PolynomialMap, x_range::AbstractVector{<:Real}, x_given::AbstractVector{<:Real})

Compute ``\\pi(x_{j:k}\\mid x_{1:j-1})``, where `x_range` contains
``x_{j:k}`` and `x_given` contains ``x_{1:j-1}``.
"""
function multivariate_conditional_density(M::PolynomialMap, x_range::AbstractVector{<:Real}, x_given::AbstractVector{<:Real})
    j = length(x_given) + 1  # Starting index for the range
    k = j + length(x_range) - 1  # Ending index for the range

    @assert j >= 1 "Starting index must be >= 1"
    @assert k <= numberdimensions(M) "Ending index cannot exceed the dimension of the map"
    @assert length(x_range) >= 1 "x_range must have at least one element"

    # Construct the full vector [x_given..., x_range...]
    x_full = [x_given..., x_range...]

    # Start with the conditional density of the first variable in the range
    density = conditional_density(M, x_range[1], x_given)

    # Multiply by conditional densities for subsequent variables in the range
    for i in 2:length(x_range)
        x_given_i = x_full[1:(j + i - 2)]  # x₁, ..., xⱼ₊ᵢ₋₂
        x_range_i = x_range[i]
        density *= conditional_density(M, x_range_i, x_given_i)
    end

    return density
end

# Convenience functions
multivariate_conditional_density(M::PolynomialMap, x_range::AbstractVector{<:Real}, x_given::Real) = multivariate_conditional_density(M, x_range, [x_given])

"""
    multivariate_conditional_sample(M::PolynomialMap, x_given::AbstractVector{<:Real}, z_range::AbstractVector{<:Real})

Generate a sample from ``\\pi(x_{j:k}\\mid x_{1:j-1})`` by sequentially
pushing the values in `z_range` through map components ``j,\\ldots,k``.
"""
function multivariate_conditional_sample(M::PolynomialMap, x_given::AbstractVector{<:Real}, z_range::AbstractVector{<:Real})
    j = length(x_given) + 1  # Starting index for sampling
    k = j + length(z_range) - 1  # Ending index for sampling

    @assert j >= 1 "Starting index must be >= 1"
    @assert k <= numberdimensions(M) "Ending index cannot exceed the dimension of the map"
    @assert length(z_range) >= 1 "z_range must have at least one element"

    x_samples = Vector{Float64}(undef, length(z_range))
    x_current = copy(x_given)

    # Generate samples sequentially using conditional sampling
    for i in 1:length(z_range)
        x_i = conditional_sample(M, x_current, z_range[i])
        x_samples[i] = x_i
        push!(x_current, x_i)  # Add to conditioning variables for next iteration
    end

    return x_samples
end

# Convenience functions
multivariate_conditional_sample(M::PolynomialMap, x_given::Real, z_range::AbstractVector{<:Real}) = multivariate_conditional_sample(M, [x_given], z_range)
