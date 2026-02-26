
# Objective for optimization of component coefficients from samples
function objective(component::PolynomialMapComponent, samples::Matrix{Float64})

    # Evaluate map component Mk and its partial derivative w.r.t. xk
    Mₖ = evaluate(component, samples)
    ∂M = partial_derivative_zk(component, samples)

    # Monte Carlo estimate of the objective
    obj = sum(0.5 * Mₖ .^ 2 - log.(abs.(∂M)))
    return obj

end

# Objective for optimization using precomputed basis evaluations
function objective(component::PolynomialMapComponent, precomp::PrecomputedBasis)
    # Get current coefficients
    c = component.coefficients

    # Evaluate Mᵏ(z) = f₀ + ∫₀^{z_k} g(∂f/∂x_k) dx_k using precomputed basis
    M_vals = evaluate_M(precomp, c, component.rectifier)

    # Evaluate ∂Mᵏ/∂zₖ = g(∂f/∂zₖ) using precomputed basis
    ∂M_vals = evaluate_∂M(precomp, c, component.rectifier)

    # Monte Carlo estimate of the objective
    obj = sum(0.5 * M_vals .^ 2 - log.(abs.(∂M_vals)))
    return obj
end

#! vectorize?
# Gradient of objective for optimization of component coefficients from samples
function objective_gradient!(Mk::PolynomialMapComponent, X::Matrix{Float64})
    # Analytical gradient of objective w.r.t. coefficients c of component Mk
    # Objective per sample: J(z) = 0.5 * Mₖ(z)^2 - log|∂Mₖ/∂zₖ(z)|
    # ∂J/∂c = Mₖ * ∂Mₖ/∂c - (1 / (∂Mₖ/∂zₖ)) * ∂(∂Mₖ/∂zₖ)/∂c

    n_coeffs = length(Mk.coefficients)
    grad = zeros(Float64, n_coeffs)

    n_points = size(X, 1)
    @inbounds for i in 1:n_points
        z = X[i, :]

        # Evaluate scalar map value and its diagonal partial derivative
        M_val = evaluate(Mk, z)
        ∂M = partial_derivative_zk(Mk, z)

        # Gradient of M w.r.t. coefficients: vector length n_coeffs
        ∂M_∂c = gradient_coefficients(Mk, z)

        # Gradient of ∂M/∂zₖ w.r.t. coefficients: vector length n_coeffs
        ∂∂M_∂c = partial_derivative_zk_gradient_coefficients(Mk, z)

        # Accumulate gradient contribution from this sample
        # Handle potential zero ∂M (should be unlikely with rectifier) by adding small eps
        denom = max(abs(∂M), eps()) * sign(∂M)

        grad .+= M_val .* ∂M_∂c .- (1.0 ./ denom) .* ∂∂M_∂c
    end

    return grad
end

# Gradient of objective using precomputed basis evaluations
function objective_gradient!(Mk::PolynomialMapComponent, precomp::PrecomputedBasis)
    # Analytical gradient of objective w.r.t. coefficients c
    # Objective: J = Σᵢ [0.5 * Mᵏ(zⁱ)² - log|∂Mᵏ/∂zₖ(zⁱ)|]
    # where Mᵏ(z) = f₀(z) + ∫₀^{z_k} g(∂f/∂x_k) dx_k

    c = Mk.coefficients
    n_samples = precomp.n_samples
    n_basis = precomp.n_basis
    n_quad = precomp.n_quad

    # Evaluate M and ∂M for all samples
    M_vals = evaluate_M(precomp, c, Mk.rectifier)
    ∂M_vals = evaluate_∂M(precomp, c, Mk.rectifier)

    # Initialize gradient
    grad = zeros(Float64, n_basis)

    # For each sample, compute gradient contributions
    @inbounds for i in 1:n_samples
        # First term: M * ∂M/∂c
        # ∂M/∂cⱼ = ∂f₀/∂cⱼ + ∂(integral)/∂cⱼ
        # where ∂f₀/∂cⱼ = ψⱼ(z₁,...,z_{k-1},0)
        #   and ∂(integral)/∂cⱼ = ∫₀^{z_k} g'(∂f/∂x_k) * ∂²f/(∂x_k∂cⱼ) dx_k
        #                        = ∫₀^{z_k} g'(∂f/∂x_k) * ∂ψⱼ/∂x_k dx_k

        # Vectorized contribution from f₀ term
        grad .+= M_vals[i] .* view(precomp.Ψ₀, i, :)

        # Contribution from integral term (using quadrature)
        scale = precomp.quad_scales[i]
        for q in 1:n_quad
            # Vectorized computation of ∂f/∂x_k at this quadrature point
            ∂f = dot(view(precomp.∂Ψ_quad, i, q, :), c)

            # Compute g'(∂f/∂x_k)
            g_prime = derivative(Mk.rectifier, ∂f)

            # Vectorized contribution: M * weight * g' * ∂ψⱼ/∂x_k
            weight_factor = M_vals[i] * precomp.quad_weights[q] * g_prime * scale
            grad .+= weight_factor .* view(precomp.∂Ψ_quad, i, q, :)
        end

        # Second term: -(1/∂M) * ∂(∂M)/∂c
        # ∂M/∂zₖ = g(∂f/∂zₖ), so ∂(∂M/∂zₖ)/∂cⱼ = g'(∂f/∂zₖ) * ∂²f/(∂zₖ∂cⱼ)
        #                                        = g'(∂f/∂zₖ) * ∂ψⱼ/∂zₖ

        # Vectorized computation of ∂f/∂zₖ at z using precomputed values
        ∂f_at_z = dot(view(precomp.∂Ψ_z, i, :), c)

        g_prime_at_z = derivative(Mk.rectifier, ∂f_at_z)
        denom = max(abs(∂M_vals[i]), eps())

        # Vectorized gradient update
        grad .-= (g_prime_at_z / denom) .* view(precomp.∂Ψ_z, i, :)
    end

    return grad
end

"""
    optimize!(M::PolynomialMap, samples::Matrix{Float64}, lm::AbstractLinearMap=LinearMap();
              optimizer::Optim.AbstractOptimizer = LBFGS(),
              options::Optim.Options = Optim.Options(),
              test_fraction::Float64 = 0.0)

Optimize polynomial map coefficients to minimize KL divergence to a target density.

# Arguments
- `M::PolynomialMap`: The polynomial map to optimize.
- `samples::Matrix{Float64}`: A matrix of sample data used to initialize and fit the map. Columns are interpreted as components/dimensions and rows as individual sample points.
- `lm::AbstractLinearMap`: A linear map used to standardize the samples before optimization (default: `LinearMap()`, identity map).

# Keyword Arguments
- `optimizer::Optim.AbstractOptimizer`: Optimizer from Optim.jl to use (default: `LBFGS()`).
- `options::Optim.Options`: Options passed to the optimizer (default: `Optim.Options()`).
- `test_fraction::Float64`: Fraction of samples to hold out for testing/validation (default: `0.0`, i.e. no test split).

# Returns
- `OptimizationResult`: Optimization results containing training and test objectives for each component. The optimized coefficients are written back into `M`.
"""
function optimize!(
    M::PolynomialMap,
    samples::Matrix{Float64},
    lm::AbstractLinearMap=LinearMap();
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options(),
    test_fraction::Float64=0.0,
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
        println("Optimizing component $(k) / $(numberdimensions(M))")

        # Precompute basis evaluations for this component
        train_precomp = PrecomputedBasis(component, train_samples[:, 1:k])
        test_precomp = !isempty(test_samples) ? PrecomputedBasis(component, test_samples[:, 1:k]) : nothing

        res = optimize!(component, train_precomp, optimizer, options)

        # Compute validation objective using precomputed basis
        train_obj = objective(component, train_precomp) / size(train_samples, 1)
        test_obj = !isnothing(test_precomp) ? objective(component, test_precomp) / size(test_samples, 1) : 0.

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

    obj_fun = c -> begin
        setcoefficients!(component, c)
        objective(component, precomp)
    end

    grad_fun! = (g, c) -> begin
        setcoefficients!(component, c)
        grad = objective_gradient!(component, precomp)
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
        train_idx = idx[n_test+1:end]
    end

    train_samples = samples[train_idx, :]
    test_samples = n_test > 0 ? samples[test_idx, :] : Array{Float64}(undef, 0, size(samples, 2))

    return train_samples, test_samples
end
