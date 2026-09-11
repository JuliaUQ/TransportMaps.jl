# Kullback-Leibler divergence between the polynomial map and a target density
function kldivergence(
        M::PolynomialMap,
        target::AbstractMapDensity,
        quadrature::AbstractQuadratureWeights,
        ; δ::Real = 1.0e-9,
    )
    @assert δ >= 0.0 "δ must be non-negative."

    # Evaluate the map and both densities in the KL change-of-variables formula.
    M_points = evaluate(M, quadrature.points) + δ * quadrature.points
    log_reference = logpdf(M.reference, quadrature.points)
    # Evaluate target logpdf
    log_π = logpdf(target, M_points)
    # Evaluate log determinant of Jacobian
    log_detJ = log.(abs.(jacobian(M, quadrature.points)))

    return sum(quadrature.weights .* (log_reference .- log_π .- log_detJ))
end

# Gradient of KL divergence with respect to map coefficients
function kldivergence_gradient(
        M::PolynomialMap,
        target::AbstractMapDensity,
        quadrature::AbstractQuadratureWeights,
        ; δ::Real = 1.0e-9,
    )
    @assert δ >= 0.0 "δ must be non-negative."

    n_coeffs = numbercoefficients(M)
    n_dims = numberdimensions(M)
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
        precomp::PrecomputedMapBasis;
        δ::Real = 1.0e-9,
    )
    @assert δ >= 0.0 "δ must be non-negative."

    # Evaluate the map and both densities in the KL change-of-variables formula.
    M_points = evaluate(M, precomp) + δ * precomp.quad_points
    log_reference = logpdf(M.reference, precomp.quad_points)
    # Evaluate target logpdf
    log_π = logpdf(target, M_points)
    # Evaluate log determinant of Jacobian
    log_detJ = log.(abs.(jacobian(M, precomp)))

    return sum(precomp.quad_weights .* (log_reference .- log_π .- log_detJ))
end

# Gradient of KL divergence using precomputed basis
function kldivergence_gradient(
        M::PolynomialMap,
        target::AbstractMapDensity,
        precomp::PrecomputedMapBasis;
        δ::Real = 1.0e-9,
    )
    @assert δ >= 0.0 "δ must be non-negative."

    n_coeffs = numbercoefficients(M)
    n_dims = numberdimensions(M)
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
    optimize!(
        M::PolynomialMap, target::AbstractMapDensity, quadrature::AbstractQuadratureWeights;
        optimizer, options, δ = 1.0e-9, λ1 = 0, λ2 = 0, l1_eps = 1.0e-8,
        interactions_only = false
    )

Optimize the polynomial map by minimizing, up to a constant,

```math
D_{\\mathrm{KL}}\\!\\left(\\rho\\,\\|\\,M^*\\pi\\right)
= \\mathbb{E}_{Z\\sim\\rho}\\!\\left[
    \\log\\rho(Z)-\\log\\pi(M(Z))-\\log|\\det\\nabla M(Z)|
  \\right].
```

The optional coefficient penalty is

```math
R(c)=\\lambda_1\\sum_{j\\in\\mathcal{I}}
\\sqrt{c_j^2+\\varepsilon^2}
+\\frac{\\lambda_2}{2}\\sum_{j\\in\\mathcal{I}}c_j^2,
```

where ``\\mathcal{I}`` contains the selected nonlinear terms.

# Arguments
- `M::PolynomialMap`: The polynomial map to optimize.
- `target::AbstractMapDensity`: Target density ``\\pi(x)``
- `quadrature::AbstractQuadratureWeights`: Quadrature points and weights used for numerical integration.

# Keyword Arguments
- `optimizer`: Optimizer from Optim.jl to use (default: `LBFGS()`).
- `options`: Options passed to the optimizer (default: `Optim.Options()`).
- `δ::Real`: Stability perturbation ``\\delta Z`` added to mapped quadrature points before target
  density evaluation (default: `1e-9`). Set it to zero for the exact KL objective.
- `λ1::Real`: Strength ``\\lambda_1`` of the smoothed L1 penalty (default: `0`).
- `λ2::Real`: Strength ``\\lambda_2`` of the L2 penalty (default: `0`).
- `l1_eps::Real`: Positive smoothing parameter used by the L1 approximation
  ``\\sqrt{a^2 + \\varepsilon^2}`` (default: `1e-8`).
- `interactions_only::Bool`: If `false`, penalize all terms of total degree two or
  greater. If `true`, penalize only terms involving at least two coordinates.

Constant and linear terms are never penalized. The L1 term is differentiable and
therefore approximates, rather than exactly equals, the L1 norm. Each penalized
coefficient contributes ``\\lambda_1\\varepsilon`` at zero; this additive constant does not affect
the minimizer.

# Returns
- Optimization result from Optim.jl. The optimized coefficients are written back into `M`.
"""
function optimize!(
        M::PolynomialMap,
        target::AbstractMapDensity,
        quadrature::AbstractQuadratureWeights;
        optimizer::Optim.AbstractOptimizer = LBFGS(),
        options::Optim.Options = Optim.Options(),
        δ::Real = 1.0e-9,
        λ1::Real = 0.0,
        λ2::Real = 0.0,
        l1_eps::Real = 1.0e-8,
        interactions_only::Bool = false,
    )
    # Precompute basis evaluations at quadrature points
    precomp = PrecomputedMapBasis(M, quadrature.points, quadrature.weights)

    # Call the optimized version
    return optimize!(
        M, target, precomp;
        optimizer = optimizer,
        options = options,
        δ = δ,
        λ1 = λ1,
        λ2 = λ2,
        l1_eps = l1_eps,
        interactions_only = interactions_only,
    )
end

# Optimized version using precomputed basis
function optimize!(
        M::PolynomialMap,
        target::AbstractMapDensity,
        precomp::PrecomputedMapBasis;
        optimizer::Optim.AbstractOptimizer = LBFGS(),
        options::Optim.Options = Optim.Options(),
        δ::Real = 1.0e-9,
        λ1::Real = 0.0,
        λ2::Real = 0.0,
        l1_eps::Real = 1.0e-8,
        interactions_only::Bool = false,
    )
    @assert λ1 >= 0.0 "λ1 must be non-negative."
    @assert λ2 >= 0.0 "λ2 must be non-negative."
    @assert l1_eps > 0.0 "l1_eps must be strictly positive."
    @assert δ >= 0.0 "δ must be non-negative."

    pen = _nonlinear_penalty_mask(M; interactions_only = interactions_only)

    function objective_function(a)
        setcoefficients!(M, a)
        loss = kldivergence(M, target, precomp; δ)
        return loss + _regularization_penalty(a, pen, λ1, λ2, l1_eps)
    end

    function gradient_function!(g, a)
        setcoefficients!(M, a)
        g .= kldivergence_gradient(M, target, precomp; δ)
        _add_regularization_gradient!(g, a, pen, λ1, λ2, l1_eps)
        return nothing
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

    @debug "Function calls" target_calls = logpdf_calls ∇target_calls = grad_logpdf_calls

    return result
end

"""
    variance_diagnostic(M::PolynomialMap, target::MapTargetDensity, Z::AbstractArray{<:Real})

Compute a variance-based diagnostic for assessing the quality of a transport map.

For reference samples ``Z_i \\sim \\rho``, the diagnostic is

```math
\\mathcal{D}_{\\mathrm{var}}(M)
= \\frac{1}{2}\\operatorname{Var}_{Z \\sim \\rho}
\\left[
    \\log \\pi\\!\\left(M(Z)\\right)
    + \\log\\left|\\det \\nabla M(Z)\\right|
    - \\log \\rho(Z)
\\right].
```

The expression in brackets is constant when ``M`` exactly pushes the reference
density ``\\rho`` forward to the target density ``\\pi``. Consequently, a value near
zero indicates a good approximation.

# Arguments
- `M::PolynomialMap`: The transport map to evaluate
- `target::MapTargetDensity`: The target density ``\\pi``
- `Z::AbstractArray{<:Real}`: Sample points from the reference distribution, where each
  row is a sample and columns correspond to dimensions

# Returns
- `Float64`: The value of ``\\mathcal{D}_{\\mathrm{var}}(M)``

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

function _nonlinear_penalty_mask(M::PolynomialMap; interactions_only::Bool = false)
    mask = Bool[]
    for component in M.components
        for α in getmultivariateindices(component)
            total_degree = sum(α)
            active_dims = count(>(0), α)

            penalize = if interactions_only
                active_dims >= 2
            else
                total_degree >= 2
            end

            push!(mask, penalize)
        end
    end
    return mask
end

function _regularization_penalty(a, pen, λ1, λ2, l1_eps)
    a_pen = a[pen]
    l1_penalty = λ1 == 0 ? zero(eltype(a)) : λ1 * sum(
            (sqrt(abs2(c) + l1_eps^2) for c in a_pen); init = zero(eltype(a)),
        )
    l2_penalty = λ2 == 0 ? zero(eltype(a)) : (λ2 / 2) * sum(abs2, a_pen; init = zero(eltype(a)))
    return l1_penalty + l2_penalty
end

function _add_regularization_gradient!(g, a, pen, λ1, λ2, l1_eps)
    if λ1 != 0 || λ2 != 0
        g[pen] .+= λ1 .* a[pen] ./ sqrt.(a[pen] .^ 2 .+ l1_eps^2) .+ λ2 .* a[pen]
    end
    return nothing
end
