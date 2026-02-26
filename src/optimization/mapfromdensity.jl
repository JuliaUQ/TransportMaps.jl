# Kullback-Leibler divergence between the polynomial map and a target density
function kldivergence(
    M::PolynomialMap,
    target::AbstractMapDensity,
    quadrature::AbstractQuadratureWeights,
)
    # Small regularization term
    δ = 1e-9

    # Evaluate map and add small δ for regularization
    M_points = evaluate(M, quadrature.points) + δ * quadrature.points
    # Evaluate target logpdf
    log_π = logpdf(target, M_points)
    # Evaluate log determinant of Jacobian
    log_detJ = log.(abs.(jacobian(M, quadrature.points)))

    return sum(quadrature.weights .* (-log_π .- log_detJ))
end

# Gradient of KL divergence with respect to map coefficients
function kldivergence_gradient(
    M::PolynomialMap,
    target::AbstractMapDensity,
    quadrature::AbstractQuadratureWeights,
)
    n_coeffs = numbercoefficients(M)
    n_dims = numberdimensions(M)
    δ = 1e-9

    # Evaluate map at all quadrature points
    M_points = evaluate(M, quadrature.points) + δ * quadrature.points

    # Evaluate gradient of target logpdf at all mapped points
    grad_logpdfs = grad_logpdf(target, M_points)

    # Compute gradient of map w.r.t. coefficients at all quadrature points
    ∂M_∂c_all = gradient_coefficients(M, quadrature.points)

    # First term: -∇_x log π(M(z)) · ∂M/∂c, weighted by quadrature
    weighted_grads = grad_logpdfs .* quadrature.weights  # (n_quad, n_dims)
    gradient_total = zeros(Float64, n_coeffs)
    for k in 1:n_dims
        gradient_total .-= vec(weighted_grads[:, k]' * ∂M_∂c_all[:, k, :])
    end

    # Second term: jacobian log-det gradient contributions
    J_contribs = jacobian_logdet_gradient(M, quadrature.points)  # (n_quad, n_coeffs)
    gradient_total .-= vec(quadrature.weights' * J_contribs)

    return gradient_total
end

# KL divergence using precomputed basis
function kldivergence(
    M::PolynomialMap,
    target::AbstractMapDensity,
    precomp::PrecomputedMapBasis
)
    δ = 1e-9  # Small value to avoid log(0)

    # Evaluate map and add small δ for regularization
    M_points = evaluate(M, precomp) + δ * precomp.quad_points
    # Evaluate target logpdf
    log_π = logpdf(target, M_points)
    # Evaluate log determinant of Jacobian
    log_detJ = log.(abs.(jacobian(M, precomp)))

    return sum(precomp.quad_weights .* (-log_π .- log_detJ))
end

# Gradient of KL divergence using precomputed basis
function kldivergence_gradient(
    M::PolynomialMap,
    target::AbstractMapDensity,
    precomp::PrecomputedMapBasis
)
    n_coeffs = numbercoefficients(M)
    n_dims = numberdimensions(M)
    δ = 1e-9

    # Evaluate map at all quadrature points
    M_points = evaluate(M, precomp) + δ * precomp.quad_points

    # Evaluate gradient of target logpdf at all mapped points
    grad_logpdfs = grad_logpdf(target, M_points)

    # Compute gradient of map w.r.t. coefficients at all quadrature points
    ∂M_∂c_all = gradient_coefficients(M, precomp)

    # First term: -∇_x log π(M(z)) · ∂M/∂c, weighted by quadrature
    weighted_grads = grad_logpdfs .* precomp.quad_weights  # (n_quad, n_dims)
    gradient_total = zeros(Float64, n_coeffs)
    for k in 1:n_dims
        gradient_total .-= vec(weighted_grads[:, k]' * ∂M_∂c_all[:, k, :])
    end

    # Second term: jacobian log-det gradient contributions
    J_contribs = jacobian_logdet_gradient(M, precomp)  # (n_quad, n_coeffs)
    gradient_total .-= vec(precomp.quad_weights' * J_contribs)
    return gradient_total
end

"""
    optimize!(M::PolynomialMap, target::AbstractMapDensity, quadrature::AbstractQuadratureWeights;
              optimizer, options)

Optimize polynomial map coefficients to minimize KL divergence to a target density.

# Arguments
- `M::PolynomialMap`: The polynomial map to optimize.
- `target::AbstractMapDensity`: Target map density object (provides the target density π(x) and any needed operations).
- `quadrature::AbstractQuadratureWeights`: Quadrature points and weights used for numerical integration.

# Keyword Arguments
- `optimizer`: Optimizer from Optim.jl to use (default: `LBFGS()`).
- `options`: Options passed to the optimizer (default: `Optim.Options()`).

# Returns
- Optimization result from Optim.jl. The optimized coefficients are written back into `M`.
"""
function optimize!(
    M::PolynomialMap,
    target::AbstractMapDensity,
    quadrature::AbstractQuadratureWeights;
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options()
)
    # Precompute basis evaluations at quadrature points
    precomp = PrecomputedMapBasis(M, quadrature.points, quadrature.weights)

    # Call the optimized version
    return optimize!(M, target, precomp, optimizer=optimizer, options=options)
end

# Optimized version using precomputed basis
function optimize!(
    M::PolynomialMap,
    target::AbstractMapDensity,
    precomp::PrecomputedMapBasis;
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options()
)

    function objective_function(a)
        setcoefficients!(M, a)
        return kldivergence(M, target, precomp)
    end

    function gradient_function!(grad_storage, a)
        setcoefficients!(M, a)
        grad_storage .= kldivergence_gradient(M, target, precomp)
    end

    initial_coefficients = getcoefficients(M)
    result = optimize(objective_function, gradient_function!, initial_coefficients, optimizer, options)

    if !Optim.converged(result)
        @warn "Optimization has not converged."
    end

    setcoefficients!(M, result.minimizer)

    # Get number of model calls
    n_quad = length(precomp.quad_weights)
    logpdf_calls = Optim.f_calls(result) * n_quad
    grad_logpdf_calls = Optim.g_calls(result) * n_quad

    @debug "Function calls" target_calls=logpdf_calls ∇target_calls=grad_logpdf_calls

    return result
end
"""
    variance_diagnostic(M::PolynomialMap, target::MapTargetDensity, Z::AbstractArray{<:Real})

Compute a variance-based diagnostic for assessing the quality of a transport map.

The diagnostic measures the variance of the log-ratio between the pushforward density
and the reference density. A smaller variance indicates a better approximation of the
target density by the transport map.

# Arguments
- `M::PolynomialMap`: The transport map to be evaluated
- `target::MapTargetDensity`: The target density that the map should approximate
- `Z::AbstractArray{<:Real}`: Sample points from the reference distribution, where each
  row is a sample and columns correspond to dimensions

# Returns
- `Float64`: The computed variance diagnostic

"""
function variance_diagnostic(
    M::PolynomialMap,
    target::MapTargetDensity,
    Z::AbstractArray{<:Real},
)
    @assert size(Z, 2) == numberdimensions(M) "Z must have the same number of columns as number of map components in M"

    log_pushforward = logpdf(target, evaluate(M, Z)) + log.(abs.(jacobian(M, Z)))
    return 0.5 * var(log_pushforward - logpdf(M.reference, Z))
end
