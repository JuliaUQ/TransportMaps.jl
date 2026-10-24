# Kullback-Leibler divergence between the polynomial map and a target density
function kldivergence(
        M::PolynomialMap,
        target::AbstractMapDensity,
        quadrature::AbstractQuadratureWeights,
        ; δ::Real = 1.0e-9,
    )
    @assert δ >= 0.0 "δ must be non-negative."

    # Evaluate the map and both densities in the KL change-of-variables formula.
    M_points = evaluate(M, quadrature.points) .+ δ .* quadrature.points
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
    M_points = evaluate(M, quadrature.points) .+ δ .* quadrature.points

    # Evaluate gradient of target logpdf at all mapped points
    grad_logpdfs = grad_logpdf(target, M_points)

    # Compute gradient of map w.r.t. coefficients at all quadrature points
    ∂M_∂c_all = gradient_coefficients(M, quadrature.points)

    # First term: -∇_x log π(M(z)) · ∂M/∂c, weighted by quadrature
    weighted_grads = grad_logpdfs .* quadrature.weights  # (n_quad, n_dims)
    gradient_total = zeros(Float64, n_coeffs)
    for k in 1:n_dims
        @views mul!(gradient_total, transpose(∂M_∂c_all[:, k, :]), weighted_grads[:, k], -1.0, 1.0)
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
    M_points = evaluate(M, precomp) .+ δ .* precomp.quad_points
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

    gradient = Vector{Float64}(undef, numbercoefficients(M))
    return _kldivergence_gradient!(gradient, M, target, precomp; δ)
end

function _kldivergence_gradient!(gradient, M, target, precomp; δ = 1.0e-9)
    M_points = evaluate(M, precomp)
    M_points .+= δ .* precomp.quad_points
    scores = grad_logpdf(target, M_points)
    offset = 0
    for k in 1:precomp.dimension
        component = M[k]
        n = length(component.coefficients)
        _density_component_gradient!(
            view(gradient, (offset + 1):(offset + n)), precomp.component_data[k],
            component.coefficients, component.rectifier,
            view(scores, :, k), precomp.quad_weights,
        )
        offset += n
    end
    return gradient
end

# Contract each component's dense coefficient block directly with target scores
# and quadrature weights. Other components' coefficient blocks are identically
# zero, so there is no need to build a (samples × dimensions × all coefficients)
# tensor or a separate log-Jacobian gradient matrix.
function _density_component_gradient!(gradient, precomp, c, rectifier, scores, weights)
    fill!(gradient, 0.0)
    map_gradient = similar(gradient)
    @inbounds for i in 1:precomp.n_samples
        _map_coefficient_gradient!(map_gradient, precomp, c, rectifier, i)
        gradient .-= (weights[i] * scores[i]) .* map_gradient

        derivative_basis = view(precomp.∂Ψ_z, i, :)
        ∂f = dot(derivative_basis, c)
        factor = weights[i] * derivative(rectifier, ∂f) / rectifier(∂f)
        gradient .-= factor .* derivative_basis
    end
    return gradient
end

function _density_workspace(M, precomp)
    return (
        components = _BasisWorkspace.(precomp.component_data),
        points = Matrix{Float64}(undef, precomp.n_quad, precomp.dimension),
        determinants = Vector{Float64}(undef, precomp.n_quad),
        log_reference = logpdf(M.reference, precomp.quad_points),
    )
end

function _density_value_gradient!(F, G, M, target, precomp, work; δ = 1.0e-9)
    fill!(work.determinants, 1.0)
    for k in 1:precomp.dimension
        component = M[k]
        component_work = work.components[k]
        _evaluate_basis!(component_work, precomp.component_data[k], component.coefficients, component.rectifier)
        @views work.points[:, k] .= component_work.values .+ δ .* precomp.quad_points[:, k]
        F === nothing || _multiply_diagonal!(work.determinants, component_work.diagonal, component.rectifier)
    end
    if G !== nothing
        scores = grad_logpdf(target, work.points)
        offset = 0
        for k in 1:precomp.dimension
            component = M[k]
            n = length(component.coefficients)
            _density_workspace_gradient!(
                view(G, (offset + 1):(offset + n)), work.components[k],
                precomp.component_data[k], component.rectifier, view(scores, :, k), precomp.quad_weights
            )
            offset += n
        end
    end
    if F !== nothing
        target_values = logpdf(target, work.points)
        return sum(
            i -> precomp.quad_weights[i] * (
                work.log_reference[i] - target_values[i] -
                    log(abs(work.determinants[i]))
            ), eachindex(precomp.quad_weights); init = 0.0
        )
    end
    return nothing
end

function _multiply_diagonal!(determinants, diagonal, rectifier)
    determinants .*= rectifier.(diagonal)
    return determinants
end

function _density_workspace_gradient!(gradient, work, precomp, rectifier, scores, weights)
    work.sample_factors .= .-weights .* scores
    _contract_basis_gradient!(gradient, work, precomp, rectifier)
    @inbounds for i in 1:precomp.n_samples
        work.sample_factors[i] = -weights[i] * derivative(rectifier, work.diagonal[i]) / rectifier(work.diagonal[i])
    end
    mul!(gradient, transpose(precomp.∂Ψ_z), work.sample_factors, 1.0, 1.0)
    return gradient
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

    work = _density_workspace(M, precomp)
    function value_gradient!(F, G, a)
        setcoefficients!(M, a)
        loss = _density_value_gradient!(F, G, M, target, precomp, work; δ)
        if G !== nothing
            _add_regularization_gradient!(G, a, pen, λ1, λ2, l1_eps)
        end
        if F !== nothing
            return loss + _regularization_penalty(a, pen, λ1, λ2, l1_eps)
        end
        return nothing
    end

    initial_coefficients = getcoefficients(M)
    objective = OnceDifferentiable(Optim.NLSolversBase.only_fg!(value_gradient!), initial_coefficients)
    result = optimize(objective, initial_coefficients, optimizer, options)

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
    penalty = zero(eltype(a))
    (iszero(λ1) && iszero(λ2)) && return penalty
    for i in eachindex(a, pen)
        pen[i] || continue
        λ1 == 0 || (penalty += λ1 * sqrt(abs2(a[i]) + l1_eps^2))
        λ2 == 0 || (penalty += (λ2 / 2) * abs2(a[i]))
    end
    return penalty
end

function _add_regularization_gradient!(g, a, pen, λ1, λ2, l1_eps)
    if λ1 != 0 || λ2 != 0
        for i in eachindex(g, a, pen)
            pen[i] || continue
            g[i] += λ1 * a[i] / sqrt(abs2(a[i]) + l1_eps^2) + λ2 * a[i]
        end
    end
    return nothing
end
