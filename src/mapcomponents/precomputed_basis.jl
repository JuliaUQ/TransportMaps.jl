"""
    PrecomputedBasis

Stores precomputed basis function evaluations for efficient optimization.

When optimizing map coefficients from samples, the basis functions are evaluated
repeatedly at the same sample points with different coefficients. This structure
precomputes and stores these evaluations to avoid redundant computation.

To evaluate

```math
M^k(z)
= f(z_1,\\ldots,z_{k-1},0)
+ \\int_0^{z_k} g\\!\\left(\\partial_k f(z_1,\\ldots,z_{k-1},t)\\right)\\,\\mathrm{d}t,
```

basis values and derivatives are precomputed at the quadrature nodes for each sample.

# Fields
- `Ψ₀::Matrix{Float64}`: Basis evaluations at ``(z_1,\\ldots,z_{k-1},0)``
- `∂Ψ_z::Matrix{Float64}`: Partial derivatives ``\\partial\\psi_j/\\partial z_k`` at the samples
- `Ψ_quad::Array{Float64,3}`: Basis evaluations at quadrature nodes
- `∂Ψ_quad::Array{Float64,3}`: Basis derivatives at quadrature nodes
- `quad_weights::Vector{Float64}`: Quadrature weights
- `quad_scales::Vector{Float64}`: Integration scale ``z_k/2`` for each sample
- `n_samples::Int`: Number of sample points
- `n_basis::Int`: Number of basis functions
- `n_quad::Int`: Number of quadrature points
"""
struct PrecomputedBasis
    Ψ₀::Matrix{Float64}              # Ψ₀[i,j] = ψⱼ(z₁,...,z_{k-1},0) for sample i
    ∂Ψ_z::Matrix{Float64}            # ∂Ψ_z[i,j] = ∂ψⱼ/∂zₖ(z) at sample point z
    Ψ_quad::Array{Float64, 3}         # Ψ_quad[i,q,j] = ψⱼ at quadrature point q for sample i
    ∂Ψ_quad::Array{Float64, 3}        # ∂Ψ_quad[i,q,j] = ∂ψⱼ/∂zₖ at quadrature point q for sample i
    quad_weights::Vector{Float64}    # Quadrature weights
    quad_scales::Vector{Float64}     # Scale factor for each sample (0.5 * z_k)
    n_samples::Int
    n_basis::Int
    n_quad::Int

    function PrecomputedBasis(
            Ψ₀::Matrix{Float64},
            ∂Ψ_z::Matrix{Float64},
            Ψ_quad::Array{Float64, 3},
            ∂Ψ_quad::Array{Float64, 3},
            quad_weights::Vector{Float64},
            quad_scales::Vector{Float64}
        )
        n_samples, n_basis = size(Ψ₀)
        n_quad = length(quad_weights)

        @assert size(∂Ψ_z) == (n_samples, n_basis) "∂Ψ_z dimensions mismatch"
        @assert size(Ψ_quad) == (n_samples, n_quad, n_basis) "Ψ_quad dimensions mismatch"
        @assert size(∂Ψ_quad) == (n_samples, n_quad, n_basis) "∂Ψ_quad dimensions mismatch"
        @assert length(quad_scales) == n_samples "quad_scales length must match n_samples"

        return new(Ψ₀, ∂Ψ_z, Ψ_quad, ∂Ψ_quad, quad_weights, quad_scales, n_samples, n_basis, n_quad)
    end
end

"""
    PrecomputedBasis(component::PolynomialMapComponent, samples::Matrix{Float64}; n_quad::Int = 64)

Precompute basis function evaluations and their partial derivatives at quadrature nodes
for all samples. This enables exact computation of the map component and its gradient
using precomputed values.

# Arguments
- `component::PolynomialMapComponent`: The map component whose basis functions to evaluate
- `samples::Matrix{Float64}`: Sample points, with one sample per row
- `n_quad::Int=64`: Number of Gauss-Legendre quadrature points (default: 64)

# Returns
- `PrecomputedBasis`: Structure containing precomputed evaluations at quadrature nodes
"""
function PrecomputedBasis(component::PolynomialMapComponent, samples::Matrix{Float64}; n_quad::Int = 64)
    n_samples = size(samples, 1)
    n_basis = length(component.basisfunctions)
    k = component.index  # The component index

    # Get Gauss-Legendre quadrature points and weights on [-1, 1]
    quad_points_std, quad_weights = gausslegendre(n_quad)

    # Preallocate arrays
    Ψ₀ = Matrix{Float64}(undef, n_samples, n_basis)
    ∂Ψ_z = Matrix{Float64}(undef, n_samples, n_basis)
    Ψ_quad = Array{Float64, 3}(undef, n_samples, n_quad, n_basis)
    ∂Ψ_quad = Array{Float64, 3}(undef, n_samples, n_quad, n_basis)
    quad_scales = Vector{Float64}(undef, n_samples)

    # Precompute for each sample
    Threads.@threads for i in 1:n_samples
        # Create thread-local buffer to avoid repeated allocations
        z_buffer = Vector{Float64}(undef, k)

        z = samples[i, :]
        z_k = z[k]

        # Scale factor for this sample: integral from 0 to z_k
        # Gauss-Legendre is on [-1,1], we map to [0, z_k]
        scale = 0.5 * z_k
        shift = 0.5 * z_k
        quad_scales[i] = scale

        # Evaluate at z with z_k = 0 for f₀ term
        # Reuse z_buffer to avoid allocation
        @inbounds for idx in 1:(k - 1)
            z_buffer[idx] = z[idx]
        end
        z_buffer[k] = 0.0

        @inbounds for j in 1:n_basis
            Ψ₀[i, j] = evaluate(component.basisfunctions[j], z_buffer)
        end

        # Evaluate ∂ψⱼ/∂zₖ at the actual sample point z
        @inbounds for j in 1:n_basis
            ∂Ψ_z[i, j] = partial_derivative_z(component.basisfunctions[j], z, k)
        end

        # Evaluate at quadrature points
        for q in 1:n_quad
            # Map quadrature point from [-1, 1] to [0, z_k]
            x_k = quad_points_std[q] * scale + shift

            # Reuse z_buffer to avoid allocation
            @inbounds for idx in 1:(k - 1)
                z_buffer[idx] = z[idx]
            end
            z_buffer[k] = x_k

            @inbounds for j in 1:n_basis
                Ψ_quad[i, q, j] = evaluate(component.basisfunctions[j], z_buffer)
                ∂Ψ_quad[i, q, j] = partial_derivative_z(component.basisfunctions[j], z_buffer, k)
            end
        end
    end

    return PrecomputedBasis(Ψ₀, ∂Ψ_z, Ψ_quad, ∂Ψ_quad, quad_weights, quad_scales)
end

"""
    evaluate_f₀(precomp::PrecomputedBasis, coefficients::Vector{Float64})

Evaluate
``f_0(z) = \\sum_j c_j\\psi_j(z_1,\\ldots,z_{k-1},0)`` for all samples.

# Arguments
- `precomp::PrecomputedBasis`: Precomputed basis evaluations
- `coefficients::Vector{Float64}`: Coefficient vector

# Returns
- `Vector{Float64}`: ``f_0`` evaluated at all sample points
"""
function evaluate_f₀(precomp::PrecomputedBasis, coefficients::Vector{Float64})
    @assert length(coefficients) == precomp.n_basis "Coefficients length must match number of basis functions"
    return precomp.Ψ₀ * coefficients
end

"""
    evaluate_integral(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)

Evaluate, for every sample,

```math
\\int_0^{z_k}
g\\!\\left(\\sum_j c_j\\,\\partial_k\\psi_j(z_1,\\ldots,z_{k-1},t)\\right)
\\,\\mathrm{d}t
```

using Gauss-Legendre quadrature with precomputed basis derivative values.

# Arguments
- `precomp::PrecomputedBasis`: Precomputed basis evaluations at quadrature nodes
- `coefficients::Vector{Float64}`: Coefficient vector
- `rectifier::AbstractRectifierFunction`: Rectifier ``g``

# Returns
- `Vector{Float64}`: Integral evaluated at all sample points (n_samples,)
"""
function evaluate_integral(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)
    @assert length(coefficients) == precomp.n_basis "Coefficients length must match number of basis functions"

    n_samples = precomp.n_samples
    n_quad = precomp.n_quad
    integrals = Vector{Float64}(undef, n_samples)
    # Vectorized computation over quadrature points for each sample
    @inbounds for i in 1:n_samples
        # Compute ∂f for all quadrature points at once: (n_quad,)
        ∂f_quad = vec(precomp.∂Ψ_quad[i, :, :] * coefficients)

        # Apply rectifier and compute weighted sum
        integral_val = dot(precomp.quad_weights, rectifier.(∂f_quad))

        # Scale by integration domain
        integrals[i] = integral_val * precomp.quad_scales[i]
    end

    return integrals
end

"""
    evaluate_M!(values, precomp, coefficients, rectifier)

Evaluate the full map component in place, writing the result into `values`.
"""
function evaluate_M!(
        values::Vector{Float64},
        precomp::PrecomputedBasis,
        coefficients::Vector{Float64},
        rectifier::AbstractRectifierFunction,
    )
    @assert length(values) == precomp.n_samples "Output length must match number of samples"
    @assert length(coefficients) == precomp.n_basis "Coefficients length must match number of basis functions"

    mul!(values, precomp.Ψ₀, coefficients)
    @inbounds for i in 1:precomp.n_samples
        integral = 0.0
        for q in 1:precomp.n_quad
            derivative_basis = view(precomp.∂Ψ_quad, i, q, :)
            ∂f = dot(derivative_basis, coefficients)
            integral += precomp.quad_weights[q] * rectifier(∂f)
        end
        values[i] += precomp.quad_scales[i] * integral
    end
    return values
end

"""
    evaluate_M(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)

Evaluate the full map component
``M^k(z) = f_0(z) + \\int_0^{z_k}g(\\partial_k f(z_{1:k-1},t))\\,\\mathrm{d}t``
for all samples.

# Arguments
- `precomp::PrecomputedBasis`: Precomputed basis evaluations
- `coefficients::Vector{Float64}`: Coefficient vector
- `rectifier::AbstractRectifierFunction`: Rectifier ``g``

# Returns
- `Vector{Float64}`: ``M^k`` evaluated at all sample points
"""
function evaluate_M(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)
    values = Vector{Float64}(undef, precomp.n_samples)
    return evaluate_M!(values, precomp, coefficients, rectifier)
end

"""
    evaluate_∂M(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)

Evaluate ``\\partial M^k/\\partial z_k = g(\\partial f/\\partial z_k)`` for all samples.

# Arguments
- `precomp::PrecomputedBasis`: Precomputed basis evaluations
- `coefficients::Vector{Float64}`: Coefficient vector
- `rectifier::AbstractRectifierFunction`: Rectifier function g

# Returns
- `Vector{Float64}`: ``\\partial M^k/\\partial z_k`` at all sample points
"""
function evaluate_∂M(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)
    @assert length(coefficients) == precomp.n_basis "Coefficients length must match number of basis functions"

    # Compute ∂f/∂zₖ at each sample point z using precomputed derivatives
    ∂f_vals = precomp.∂Ψ_z * coefficients

    # Apply rectifier
    return rectifier.(∂f_vals)
end

# Display methods
function Base.show(io::IO, pb::PrecomputedBasis)
    print(io, "PrecomputedBasis($(pb.n_samples) samples, $(pb.n_basis) basis functions, $(pb.n_quad) quad points)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", pb::PrecomputedBasis)
    println(io, "PrecomputedBasis:")
    println(io, "  Number of samples: $(pb.n_samples)")
    println(io, "  Number of basis functions: $(pb.n_basis)")
    println(io, "  Number of quadrature points: $(pb.n_quad)")

    # Memory usage estimate
    memory_mb = (
        sizeof(pb.Ψ₀) + sizeof(pb.∂Ψ_z) + sizeof(pb.Ψ_quad) + sizeof(pb.∂Ψ_quad) +
            sizeof(pb.quad_weights) + sizeof(pb.quad_scales)
    ) / (1024^2)
    println(io, "  Memory usage: $(round(memory_mb, digits = 2)) MB")
    return nothing
end

"""
    PrecomputedMapBasis

Stores precomputed basis evaluations for a full PolynomialMap at quadrature points.

Used for density-based optimization where the same quadrature points are used throughout
optimization with different coefficient values.

# Fields
- `component_data::Vector{PrecomputedBasis}`: Precomputed data for each component
- `quad_points::Matrix{Float64}`: Quadrature points (n_quad × dimension)
- `quad_weights::Vector{Float64}`: Quadrature weights
- `n_quad::Int`: Number of quadrature points
- `dimension::Int`: Number of map components/dimensions
"""
struct PrecomputedMapBasis
    component_data::Vector{PrecomputedBasis}
    quad_points::Matrix{Float64}
    quad_weights::Vector{Float64}
    n_quad::Int
    dimension::Int

    function PrecomputedMapBasis(
            component_data::Vector{PrecomputedBasis},
            quad_points::Matrix{Float64},
            quad_weights::Vector{Float64}
        )
        n_quad = length(quad_weights)
        dimension = length(component_data)

        @assert size(quad_points, 1) == n_quad "quad_points rows must match n_quad"
        @assert size(quad_points, 2) == dimension "quad_points columns must match dimension"

        return new(component_data, quad_points, quad_weights, n_quad, dimension)
    end
end

"""
    PrecomputedMapBasis(M::PolynomialMap, quad_points::Matrix{Float64}, quad_weights::Vector{Float64}; n_quad::Int = 64)

Precompute basis evaluations for all components of a PolynomialMap at given quadrature points.

# Arguments
- `M::PolynomialMap`: The polynomial map
- `quad_points::Matrix{Float64}`: Quadrature points (n_quad × dimension)
- `quad_weights::Vector{Float64}`: Quadrature weights (n_quad,)
- `n_quad::Int=64`: Number of internal quadrature points for component integrals

# Returns
- `PrecomputedMapBasis`: Structure containing precomputed evaluations for all components
"""
function PrecomputedMapBasis(
        M::PolynomialMap,
        quad_points::Matrix{Float64},
        quad_weights::Vector{Float64};
        n_quad::Int = 64
    )
    dimension = numberdimensions(M)
    n_quad_pts = size(quad_points, 1)

    @assert size(quad_points, 2) == dimension "Quadrature points must match map dimension"
    @assert length(quad_weights) == n_quad_pts "Weights must match number of quadrature points"

    # Precompute basis for each component
    component_data = Vector{PrecomputedBasis}(undef, dimension)

    for k in 1:dimension
        component = M[k]
        # Extract relevant columns (z₁, ..., zₖ) from quadrature points
        samples_k = quad_points[:, 1:k]
        component_data[k] = PrecomputedBasis(component, samples_k, n_quad = n_quad)
    end

    return PrecomputedMapBasis(component_data, quad_points, quad_weights)
end

# Display methods
function Base.show(io::IO, pmb::PrecomputedMapBasis)
    print(io, "PrecomputedMapBasis($(pmb.n_quad) quadrature points, $(pmb.dimension) dimensions)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", pmb::PrecomputedMapBasis)
    println(io, "PrecomputedMapBasis:")
    println(io, "  Number of quadrature points: $(pmb.n_quad)")
    println(io, "  Map dimensions: $(pmb.dimension)")

    # Memory usage estimate
    total_memory = sizeof(pmb.quad_points) + sizeof(pmb.quad_weights)
    for comp_data in pmb.component_data
        total_memory += (
            sizeof(comp_data.Ψ₀) + sizeof(comp_data.∂Ψ_z) +
                sizeof(comp_data.Ψ_quad) + sizeof(comp_data.∂Ψ_quad) +
                sizeof(comp_data.quad_weights) + sizeof(comp_data.quad_scales)
        )
    end
    memory_mb = total_memory / (1024^2)
    println(io, "  Memory usage: $(round(memory_mb, digits = 2)) MB")
    return nothing
end

function evaluate(M::PolynomialMap, precomp::PrecomputedMapBasis)
    dimension = precomp.dimension
    n_quad = precomp.n_quad
    result = Matrix{Float64}(undef, n_quad, dimension)

    # Preallocate temporary arrays for batch computation
    f₀_all = Matrix{Float64}(undef, n_quad, dimension)
    integral_all = Matrix{Float64}(undef, n_quad, dimension)

    @inbounds for k in 1:dimension
        component = M[k]
        comp_precomp = precomp.component_data[k]
        c = component.coefficients

        # Vectorized f₀ computation: matrix-vector multiplication
        # f₀_all[:, k] = comp_precomp.Ψ₀ * c
        mul!(view(f₀_all, :, k), comp_precomp.Ψ₀, c)

        # Integral term computation
        Threads.@threads for i in 1:n_quad
            scale = comp_precomp.quad_scales[i]
            integral_val = 0.0

            # Vectorized computation over quadrature points
            for q in 1:comp_precomp.n_quad
                ∂f = dot(view(comp_precomp.∂Ψ_quad, i, q, :), c)
                integral_val += comp_precomp.quad_weights[q] * component.rectifier(∂f)
            end

            integral_all[i, k] = integral_val * scale
        end

        # Combine f₀ and integral terms
        result[:, k] .= f₀_all[:, k] .+ integral_all[:, k]
    end

    return result
end

function gradient_coefficients(M::PolynomialMap, precomp::PrecomputedMapBasis)
    dimension = precomp.dimension
    n_quad = precomp.n_quad
    n_total_coeffs = numbercoefficients(M)

    grad_matrix = zeros(Float64, n_quad, dimension, n_total_coeffs)

    coeff_offset = 1
    @inbounds for k in 1:dimension
        component = M[k]
        comp_precomp = precomp.component_data[k]
        c = component.coefficients
        n_comp_coeffs = length(c)

        # For each coefficient of this component
        for j in 1:n_comp_coeffs
            Threads.@threads for i in 1:n_quad
                # ∂M^k/∂c_j = ∂f₀/∂c_j + ∂(integral)/∂c_j

                # f₀ term contribution
                grad_val = comp_precomp.Ψ₀[i, j]

                # Integral term contribution
                # ∂(integral)/∂c_j = ∫₀^{z_k} g'(∂f/∂x_k) * ∂²f/(∂x_k ∂c_j) dx_k
                #                  = ∫₀^{z_k} g'(∂f/∂x_k) * ∂ψ_j/∂x_k dx_k
                scale = comp_precomp.quad_scales[i]
                for q in 1:comp_precomp.n_quad
                    # Vectorized computation of ∂f/∂x_k at this quadrature point
                    ∂f = dot(view(comp_precomp.∂Ψ_quad, i, q, :), c)

                    g_prime = derivative(component.rectifier, ∂f)
                    grad_val += comp_precomp.quad_weights[q] * g_prime * comp_precomp.∂Ψ_quad[i, q, j] * scale
                end

                grad_matrix[i, k, coeff_offset + j - 1] = grad_val
            end
        end

        coeff_offset += n_comp_coeffs
    end

    return grad_matrix
end

function jacobian(M::PolynomialMap, precomp::PrecomputedMapBasis)
    dimension = precomp.dimension
    n_quad = precomp.n_quad
    diag = Matrix{Float64}(undef, n_quad, dimension)

    @inbounds for k in 1:dimension
        component = M[k]
        comp_precomp = precomp.component_data[k]
        c = component.coefficients

        Threads.@threads for i in 1:n_quad
            # ∂M^k/∂z_k = g(∂f/∂z_k)
            # Vectorized computation of ∂f/∂z_k at the quadrature point
            ∂f = dot(view(comp_precomp.∂Ψ_z, i, :), c)

            diag[i, k] = component.rectifier(∂f)
        end
    end

    return prod.(eachrow(diag))
end

function jacobian_logdet_gradient(M::PolynomialMap, precomp::PrecomputedMapBasis)
    dimension = precomp.dimension
    n_quad = precomp.n_quad
    n_total_coeffs = numbercoefficients(M)

    gradient = zeros(Float64, n_quad, n_total_coeffs)

    coeff_offset = 1
    @inbounds for k in 1:dimension
        component = M[k]
        comp_precomp = precomp.component_data[k]
        c = component.coefficients
        n_comp_coeffs = length(c)

        Threads.@threads for i in 1:n_quad
            # Vectorized computation of ∂M^k/∂z_k (diagonal element)
            ∂f = dot(view(comp_precomp.∂Ψ_z, i, :), c)

            diagonal_deriv = component.rectifier(∂f)
            g_prime = derivative(component.rectifier, ∂f)

            # ∂(log|∂M^k/∂z_k|)/∂c_j = (1/∂M^k/∂z_k) * g'(∂f/∂z_k) * ∂ψ_j/∂z_k
            # Vectorized assignment of gradient components
            gradient[i, coeff_offset:(coeff_offset + n_comp_coeffs - 1)] .= (g_prime / diagonal_deriv) .* view(comp_precomp.∂Ψ_z, i, :)
        end

        coeff_offset += n_comp_coeffs
    end

    return gradient
end
