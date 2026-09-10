# The two-argument methods retain the original standard-normal default.
objective(component::PolynomialMapComponent, samples::Matrix{Float64}) =
    objective(component, MapReferenceDensity(), samples)

objective(component::PolynomialMapComponent, precomp::PrecomputedBasis) =
    objective(component, MapReferenceDensity(), precomp)

objective_gradient!(component::PolynomialMapComponent, samples::Matrix{Float64}) =
    objective_gradient!(component, MapReferenceDensity(), samples)

objective_gradient!(component::PolynomialMapComponent, precomp::PrecomputedBasis) =
    objective_gradient!(component, MapReferenceDensity(), precomp)

# Reference-aware sample objective. The two-argument methods above are kept for
# backwards compatibility and retain their standard-normal default.
function objective(
        component::PolynomialMapComponent,
        reference::MapReferenceDensity,
        samples::Matrix{Float64},
    )
    M_vals = evaluate(component, samples)
    ∂M_vals = partial_derivative_zk(component, samples)
    return -logpdf(reference, M_vals) - sum(log.(abs.(∂M_vals)))
end

function objective(
        component::PolynomialMapComponent,
        reference::MapReferenceDensity,
        precomp::PrecomputedBasis,
    )
    c = component.coefficients
    M_vals = evaluate_M(precomp, c, component.rectifier)
    ∂M_vals = evaluate_∂M(precomp, c, component.rectifier)
    return -logpdf(reference, M_vals) - sum(log.(abs.(∂M_vals)))
end

function objective_gradient!(
        component::PolynomialMapComponent,
        reference::MapReferenceDensity,
        samples::Matrix{Float64},
    )
    M_vals = evaluate(component, samples)
    negative_scores = -grad_logpdf(reference, M_vals)
    grad = zeros(Float64, length(component.coefficients))

    @inbounds for i in axes(samples, 1)
        z = samples[i, :]
        ∂M = partial_derivative_zk(component, z)
        ∂M_∂c = gradient_coefficients(component, z)
        ∂∂M_∂c = partial_derivative_zk_gradient_coefficients(component, z)
        denom = max(abs(∂M), eps()) * (signbit(∂M) ? -1.0 : 1.0)

        grad .+= negative_scores[i] .* ∂M_∂c .- (1.0 / denom) .* ∂∂M_∂c
    end
    return grad
end

# Write ∂Mᵏ(xⁱ)/∂c for one sample into `gradient`.
function _map_coefficient_gradient!(
        gradient::AbstractVector{Float64},
        component::PolynomialMapComponent,
        precomp::PrecomputedBasis,
        sample_index::Int,
    )
    c = component.coefficients
    copyto!(gradient, view(precomp.Ψ₀, sample_index, :))
    scale = precomp.quad_scales[sample_index]

    @inbounds for q in 1:precomp.n_quad
        ∂f = dot(view(precomp.∂Ψ_quad, sample_index, q, :), c)
        weight = precomp.quad_weights[q] * derivative(component.rectifier, ∂f) * scale
        gradient .+= weight .* view(precomp.∂Ψ_quad, sample_index, q, :)
    end
    return gradient
end

# Return ∂Mᵏ(xⁱ)/∂c for one sample from a precomputed basis.
function _map_coefficient_gradient(
        component::PolynomialMapComponent,
        precomp::PrecomputedBasis,
        sample_index::Int,
    )
    gradient = Vector{Float64}(undef, precomp.n_basis)
    return _map_coefficient_gradient!(gradient, component, precomp, sample_index)
end

function objective_gradient!(
        component::PolynomialMapComponent,
        reference::MapReferenceDensity,
        precomp::PrecomputedBasis,
    )
    c = component.coefficients
    M_vals = evaluate_M(precomp, c, component.rectifier)
    ∂M_vals = evaluate_∂M(precomp, c, component.rectifier)
    negative_scores = -grad_logpdf(reference, M_vals)
    grad = zeros(Float64, precomp.n_basis)
    map_gradient = similar(grad)

    @inbounds for i in 1:precomp.n_samples
        _map_coefficient_gradient!(map_gradient, component, precomp, i)
        grad .+= negative_scores[i] .* map_gradient

        ∂f_at_z = dot(view(precomp.∂Ψ_z, i, :), c)
        g_prime_at_z = derivative(component.rectifier, ∂f_at_z)
        denom = max(abs(∂M_vals[i]), eps()) * (signbit(∂M_vals[i]) ? -1.0 : 1.0)
        grad .-= (g_prime_at_z / denom) .* view(precomp.∂Ψ_z, i, :)
    end
    return grad
end

_default_sample_optimizer(reference::MapReferenceDensity) =
    reference.densitytype isa Uniform ? IPNewton() : LBFGS()

function _inverse_rectifier(rectifier::Softplus, value::Real)
    @assert value > 0 "The requested map slope must be positive."
    βvalue = rectifier.β * value
    return value + log(-expm1(-βvalue)) / rectifier.β
end

function _inverse_rectifier(::ShiftedELU, value::Real)
    @assert value > 0 "The requested map slope must be positive."
    return value <= 1 ? log(value) : value - 1
end


function _inverse_rectifier(::ExpRectifier, value::Real)
    @assert value > 0 "The requested map slope must be positive."
    return log(value)
end


_inverse_rectifier(::IdentityRectifier, value::Real) = value

# Construct an affine, strictly feasible starting map for a uniform reference.
function _initialize_uniform_component!(
        component::PolynomialMapComponent,
        samples::Matrix{Float64},
        reference::Uniform,
    )
    lower, upper = extrema(reference)
    width = upper - lower
    sample_lower, sample_upper = extrema(view(samples, :, component.index))
    sample_width = sample_upper - sample_lower
    @assert sample_width > 0 "Cannot fit a map to a uniform reference when a sample dimension is constant."

    initial_margin = 0.05 * width
    slope = (width - 2 * initial_margin) / sample_width
    intercept = lower + initial_margin - slope * sample_lower

    multiindices = getmultivariateindices(component)
    constant_index = findfirst(==(zeros(Int, component.index)), multiindices)
    linear_multiindex = zeros(Int, component.index)
    linear_multiindex[end] = 1
    linear_index = findfirst(==(linear_multiindex), multiindices)
    @assert !isnothing(constant_index) "A constant basis term is required for uniform-reference fitting."
    @assert !isnothing(linear_index) "A diagonal linear basis term is required for uniform-reference fitting."

    coefficients = zeros(length(component.coefficients))
    origin = zeros(component.index)
    basis_derivative = partial_derivative_z(
        component.basisfunctions[linear_index], origin, component.index
    )
    @assert !iszero(basis_derivative) "The diagonal linear basis term must have a nonzero derivative."
    coefficients[linear_index] = _inverse_rectifier(component.rectifier, slope) / basis_derivative

    constant_value = evaluate(component.basisfunctions[constant_index], origin)
    value_at_origin = f(component.basisfunctions, coefficients, origin)
    coefficients[constant_index] += (intercept - value_at_origin) / constant_value
    setcoefficients!(component, coefficients)
    return nothing
end

"""
    optimize!(
        M::PolynomialMap, samples::Matrix{Float64}, lm::AbstractLinearMap = LinearMap();
        optimizer::Optim.AbstractOptimizer = _default_sample_optimizer(M.reference),
        options::Optim.Options = Optim.Options(),
        test_fraction::Float64 = 0.0
    )

Fit the inverse transport ``S:x\\mapsto z`` by minimizing the empirical objective

```math
\\widehat J(S)
= -\\frac{1}{N}\\sum_{i=1}^N
  \\left[
    \\log\\rho(S(x_i))
    +\\log|\\det\\nabla S(x_i)|
  \\right].
```

For a triangular map, this separates into one optimization problem per component.
If ``\\rho`` is uniform on ``[a,b]^d``, the fit additionally enforces
``a<S^k(x_i)<b`` at every training sample.

# Arguments
- `M::PolynomialMap`: The polynomial map to optimize.
- `samples::Matrix{Float64}`: A matrix of sample data used to initialize and fit the map. Columns are interpreted as components/dimensions and rows as individual sample points.
- `lm::AbstractLinearMap`: A linear map used to standardize the samples before optimization (default: `LinearMap()`, identity map).

# Keyword Arguments
- `optimizer::Optim.AbstractOptimizer`: Optimizer from Optim.jl to use. Defaults to
  `LBFGS()` for unbounded references and `IPNewton()` for a uniform reference.
- `options::Optim.Options`: Options passed to the optimizer (default: `Optim.Options()`).
- `test_fraction::Float64`: Fraction of samples to hold out for testing/validation (default: `0.0`, i.e. no test split).

# Returns
- `OptimizationResult`: Optimization results containing training and test objectives for each component. The optimized coefficients are written back into `M`.
"""
function optimize!(
        M::PolynomialMap,
        samples::Matrix{Float64},
        lm::AbstractLinearMap = LinearMap();
        optimizer::Optim.AbstractOptimizer = _default_sample_optimizer(M.reference),
        options::Optim.Options = Optim.Options(),
        test_fraction::Float64 = 0.0,
    )
    @assert size(samples, 2) == numberdimensions(M) "Samples must have the same number of columns as number of map components in M"
    # Standardize samples using linear map
    samples = evaluate(lm, samples)

    # Initialize map from samples: set map direction and bounds (use full samples)
    initializemapfromsamples!(M, samples)

    # Prepare train/test split
    train_samples, test_samples = _test_train_split(samples, test_fraction)

    # Store optimization results
    results = OptimizationResult(numberdimensions(M))

    # Optimize each component sequentially using the training split
    for k in 1:numberdimensions(M)
        component = M[k]
        @debug "Optimizing component $(k) / $(numberdimensions(M))"

        component_samples = train_samples[:, 1:k]
        if M.reference.densitytype isa Uniform
            _initialize_uniform_component!(component, component_samples, M.reference.densitytype)
        end

        # Precompute basis evaluations for this component
        train_precomp = PrecomputedBasis(component, component_samples)
        test_precomp = !isempty(test_samples) ? PrecomputedBasis(component, test_samples[:, 1:k]) : nothing

        res = optimize!(component, M.reference, train_precomp, optimizer, options)

        # Compute validation objective using precomputed basis
        train_obj = objective(component, M.reference, train_precomp) / size(train_samples, 1)
        test_obj = !isnothing(test_precomp) ? objective(component, M.reference, test_precomp) / size(test_samples, 1) : 0.0

        update_optimization_result!(results, k, train_obj, test_obj, res)
    end

    return results
end

# Optimize a single map component (original interface, for backwards compatibility)
function optimize!(
        component::PolynomialMapComponent,
        samples::Matrix{Float64},
        optimizer::Optim.AbstractOptimizer,
        options::Optim.Options
    )
    # Precompute basis evaluations
    precomp = PrecomputedBasis(component, samples)

    # Call the optimized version
    return optimize!(component, precomp, optimizer, options)
end

# Optimize a single map component using precomputed basis
function optimize!(
        component::PolynomialMapComponent,
        precomp::PrecomputedBasis,
        optimizer::Optim.AbstractOptimizer,
        options::Optim.Options
    )
    return optimize!(
        component, MapReferenceDensity(), precomp, optimizer, options
    )
end

function optimize!(
        component::PolynomialMapComponent,
        reference::MapReferenceDensity,
        precomp::PrecomputedBasis,
        optimizer::Optim.AbstractOptimizer,
        options::Optim.Options,
    )
    if reference.densitytype isa Uniform
        return _optimize_uniform_component!(
            component, reference, precomp, optimizer, options
        )
    end

    obj_fun = c -> begin
        setcoefficients!(component, c)
        objective(component, reference, precomp)
    end

    grad_fun! = (g, c) -> begin
        setcoefficients!(component, c)
        grad = objective_gradient!(component, reference, precomp)
        g .= grad
    end

    initial_coefficients = getcoefficients(component)
    result = optimize(obj_fun, grad_fun!, initial_coefficients, optimizer, options)

    if !Optim.converged(result)
        @warn "Optimization has not converged."
    end

    # Update map component with optimized coefficients
    setcoefficients!(component, result.minimizer)

    return result
end

function _uniform_objective_hessian!(
        hessian::Matrix{Float64},
        component::PolynomialMapComponent,
        precomp::PrecomputedBasis,
        coefficients::Vector{Float64},
    )
    fill!(hessian, 0.0)
    @inbounds for i in 1:precomp.n_samples
        derivative_basis = view(precomp.∂Ψ_z, i, :)
        ∂f = dot(derivative_basis, coefficients)
        diagonal_derivative = component.rectifier(∂f)
        rectifier_derivative = derivative(component.rectifier, ∂f)
        rectifier_second_derivative = second_derivative(component.rectifier, ∂f)
        curvature =
            (rectifier_derivative / diagonal_derivative)^2 -
            rectifier_second_derivative / diagonal_derivative
        BLAS.ger!(curvature, derivative_basis, derivative_basis, hessian)
    end
    return hessian
end

function _uniform_constraint_hessian!(
        hessian::Matrix{Float64},
        component::PolynomialMapComponent,
        precomp::PrecomputedBasis,
        coefficients::Vector{Float64},
        multipliers::Vector{Float64},
    )
    @assert length(multipliers) == precomp.n_samples "Multiplier length must match number of samples"
    fill!(hessian, 0.0)
    @inbounds for i in 1:precomp.n_samples
        sample_scale = multipliers[i] * precomp.quad_scales[i]
        iszero(sample_scale) && continue
        for q in 1:precomp.n_quad
            derivative_basis = view(precomp.∂Ψ_quad, i, q, :)
            ∂f = dot(derivative_basis, coefficients)
            curvature =
                sample_scale * precomp.quad_weights[q] *
                second_derivative(component.rectifier, ∂f)
            BLAS.ger!(curvature, derivative_basis, derivative_basis, hessian)
        end
    end
    return hessian
end

function _optimize_uniform_component!(
        component::PolynomialMapComponent,
        reference::MapReferenceDensity,
        precomp::PrecomputedBasis,
        optimizer::Optim.AbstractOptimizer,
        options::Optim.Options,
    )
    if !(optimizer isa IPNewton)
        throw(
            ArgumentError(
                "A uniform reference has bounded support and requires the constrained " *
                    "IPNewton optimizer. Omit `optimizer` or pass `optimizer=IPNewton()`."
            )
        )
    end

    distribution = reference.densitytype
    lower, upper = extrema(distribution)
    width = upper - lower
    support_margin = 0.01 * width
    constraint_lower = lower + support_margin
    constraint_upper = upper - support_margin
    n_coefficients = length(component.coefficients)

    obj_fun = c -> begin
        setcoefficients!(component, c)
        ∂M_vals = evaluate_∂M(precomp, c, component.rectifier)
        precomp.n_samples * log(width) - sum(log.(abs.(∂M_vals)))
    end

    grad_fun! = (g, c) -> begin
        setcoefficients!(component, c)
        g .= objective_gradient!(component, reference, precomp)
    end

    constraint_fun! = (values, c) -> begin
        evaluate_M!(values, precomp, c, component.rectifier)
    end

    objective_hessian! = (hessian, c) -> begin
        _uniform_objective_hessian!(hessian, component, precomp, c)
    end

    constraint_jacobian! = (jacobian, c) -> begin
        setcoefficients!(component, c)
        @inbounds for i in 1:precomp.n_samples
            _map_coefficient_gradient!(
                view(jacobian, i, :), component, precomp, i
            )
        end
    end

    constraint_hessian! = (hessian, c, multipliers) -> begin
        _uniform_constraint_hessian!(
            hessian, component, precomp, c, multipliers
        )
    end

    initial_coefficients = getcoefficients(component)
    objective_twice_differentiable = TwiceDifferentiable(
        obj_fun, grad_fun!, objective_hessian!, initial_coefficients
    )
    constraints = TwiceDifferentiableConstraints(
        constraint_fun!,
        constraint_jacobian!,
        constraint_hessian!,
        fill(-Inf, n_coefficients),
        fill(Inf, n_coefficients),
        fill(constraint_lower, precomp.n_samples),
        fill(constraint_upper, precomp.n_samples),
    )
    result = optimize(
        objective_twice_differentiable,
        constraints,
        initial_coefficients,
        optimizer,
        options,
    )

    if !Optim.converged(result)
        @warn "Optimization has not converged."
    end

    setcoefficients!(component, result.minimizer)
    return result
end

# Helper function to create train/test split
function _test_train_split(samples::Matrix{Float64}, test_fraction::Float64)
    @assert 0.0 <= test_fraction < 1.0 "test_fraction must be in [0, 1)"
    n_points = size(samples, 1)

    n_test = test_fraction == 0.0 ? 0 : max(1, round(Int, test_fraction * n_points))

    train_idx = collect(1:n_points)
    test_idx = Int[]
    if n_test > 0
        idx = randperm(n_points)
        test_idx = idx[1:n_test]
        train_idx = idx[(n_test + 1):end]
    end

    train_samples = samples[train_idx, :]
    test_samples = n_test > 0 ? samples[test_idx, :] : Array{Float64}(undef, 0, size(samples, 2))

    return train_samples, test_samples
end
