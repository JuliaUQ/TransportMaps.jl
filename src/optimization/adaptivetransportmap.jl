# Adaptive transport map optimization based on greedy basis selection
# as described in:
#   Baptista, R., Marzouk, Y., & Zahm, O. (2023). On the Representation and Learning of
#   Monotone Triangular Transport Maps. Foundations of Computational Mathematics.
#   https://doi.org/10.1007/s10208-023-09630-x

"""
    optimize_adaptive_transportmap(samples, maxterms, lm, reference, rectifier, basis; kwargs...)

Adaptively optimize a triangular transport map by greedily enriching the multi-index set.

# Arguments
- `samples::Matrix{Float64}`: Sample data where rows are samples and columns are dimensions
- `maxterms::Vector{Int64}`: Maximum number of terms for each component
- `lm::AbstractLinearMap=LinearMap(samples)`: Linear map to standardize samples
- `reference::Distributions.UnivariateDistribution=Normal()`: Reference distribution
- `rectifier::AbstractRectifierFunction=Softplus()`: Rectifier function to use
- `basis::AbstractPolynomialBasis=_default_basis(reference)`: Polynomial basis selected from the reference

# Keyword Arguments
- `optimizer::Optim.AbstractOptimizer`: Defaults to `IPNewton()` for uniform references and `LBFGS()` otherwise
- `options::Optim.Options=Optim.Options()`: Optimizer options
- `test_fraction::Float64=0.2`: Fraction of samples to use for testing (validation)

Uniform references require at least two terms per component and start with a constant
and a diagonal linear term. Each history entry records the terms used in that iteration.

# Returns
- `C::ComposedMap`: The optimized triangular transport map
- `iteration_histories::Vector{OptimizationHistory}`: History of optimization for each component
"""
function optimize_adaptive_transportmap(
        samples::Matrix{Float64},
        maxterms::Vector{Int64},
        lm::AbstractLinearMap = LinearMap(samples),
        reference::Distributions.UnivariateDistribution = Normal(),
        rectifier::AbstractRectifierFunction = Softplus(),
        basis::AbstractPolynomialBasis = _default_basis(reference);
        optimizer::Optim.AbstractOptimizer = _default_sample_optimizer(MapReferenceDensity(reference)),
        options::Optim.Options = Optim.Options(),
        test_fraction::Float64 = 0.2
    )
    @assert length(maxterms) == size(samples, 2) "Length of maxterms must equal number of dimensions in samples"
    @assert validate_basis_reference_compatibility(basis, reference) "Invalid combination of reference density and basis."
    # Extract number of dimensions from samples
    d = size(samples, 2)
    T = typeof(basis)
    map_components = Vector{PolynomialMapComponent{T}}()
    iteration_histories = Vector{OptimizationHistory}()

    # Standardize samples using linear map
    samples = evaluate(lm, samples)

    # Prepare train/test split
    train_samples, test_samples = _test_train_split(samples, test_fraction)

    for k in 1:d
        @info "Starting adaptive component optimization" component = k maxterms = maxterms[k]
        component, history = optimize_adaptive_transportmapcomponent(
            train_samples[:, 1:k],
            test_samples[:, 1:k],
            maxterms[k],
            reference,
            rectifier,
            basis;
            optimizer,
            options,
        )
        push!(map_components, component)
        push!(iteration_histories, history)
    end

    # Construct final map from optimized components
    M = PolynomialMap(map_components, reference; forwarddirection = :reference)

    # Return composed map (including linear standardization) and histories
    return ComposedMap(lm, M), iteration_histories
end

"""
    optimize_adaptive_transportmap(samples, maxterms, k_folds, lm, reference, rectifier, basis; optimizer, options)

Adaptively optimize a triangular transport map using k-fold cross-validation to
select the number of terms per component.

# Arguments
- `samples::Matrix{Float64}`: Sample data where rows are samples and columns are dimensions
- `maxterms::Vector{Int64}`: Upper bound on the number of terms to consider for each component
- `k_folds::Int`: Number of folds to use for cross-validation
- `lm::AbstractLinearMap`: Linear map to standardize samples (default: `LinearMap(samples)`)
- `reference::Distributions.UnivariateDistribution`: Reference distribution (default: `Normal()`)
- `rectifier::AbstractRectifierFunction`: Rectifier function to use (default: `Softplus()`)
- `basis::AbstractPolynomialBasis`: Polynomial basis (default: `_default_basis(reference)`)

# Keyword Arguments
- `optimizer::Optim.AbstractOptimizer`: Defaults to `IPNewton()` for uniform references and `LBFGS()` otherwise
- `options::Optim.Options=Optim.Options()`: Optimizer options

# Returns
- `ComposedMap`: The optimized triangular transport map trained on all samples
- `fold_histories::Vector{Vector{OptimizationHistory}}`: Optimization history for each component and fold
- `selected_terms::Vector{Int}`: Number of terms selected for each component
- `selected_fold::Vector{Int}`: Index of the best fold for each component
"""
function optimize_adaptive_transportmap(
        samples::Matrix{Float64},
        maxterms::Vector{Int64},
        k_folds::Int,
        lm::AbstractLinearMap = LinearMap(samples),
        reference::Distributions.UnivariateDistribution = Normal(),
        rectifier::AbstractRectifierFunction = Softplus(),
        basis::AbstractPolynomialBasis = _default_basis(reference);
        optimizer::Optim.AbstractOptimizer = _default_sample_optimizer(MapReferenceDensity(reference)),
        options::Optim.Options = Optim.Options()
    )
    @assert length(maxterms) == size(samples, 2) "Length of maxterms must equal number of dimensions in samples"
    @assert validate_basis_reference_compatibility(basis, reference) "Invalid combination of reference density and basis."
    @assert k_folds >= 2 "k_folds must be at least 2"
    @assert k_folds <= size(samples, 1) "k_folds cannot exceed the number of samples"

    d = size(samples, 2)
    T = typeof(basis)
    map_components = Vector{PolynomialMapComponent{T}}(undef, d)
    fold_histories = Vector{Vector{OptimizationHistory}}(undef, d)
    selected_fold = Vector{Int}(undef, d)
    selected_terms = Vector{Int}(undef, d)

    samples = evaluate(lm, samples)
    folds = _kfold_indices(size(samples, 1), k_folds)

    for k in 1:d
        @info "Starting k-fold adaptive component optimization" component = k folds = k_folds maxterms = maxterms[k]
        component_histories = Vector{OptimizationHistory}(undef, k_folds)

        for fold_id in 1:k_folds
            test_idx = folds[fold_id]
            train_idx = [idx for (j, idxs) in enumerate(folds) if j != fold_id for idx in idxs]

            train_samples_fold = samples[train_idx, 1:k]
            test_samples_fold = samples[test_idx, 1:k]

            @debug "Optimizing cross-validation fold" component = k fold = fold_id folds = k_folds
            _, history_fold = optimize_adaptive_transportmapcomponent(
                train_samples_fold,
                test_samples_fold,
                maxterms[k],
                reference,
                rectifier,
                basis;
                optimizer,
                options,
            )

            component_histories[fold_id] = history_fold
        end

        mean_test_objectives = [Statistics.mean(history.test_objectives[t] for history in component_histories) for t in eachindex(component_histories[1].terms)]
        best_iteration = argmin(mean_test_objectives)
        selected_terms[k] = size(component_histories[1].terms[best_iteration], 1)

        fold_scores = [component_histories[i].test_objectives[best_iteration] for i in 1:k_folds]

        best_fold = argmin(fold_scores)
        selected_fold[k] = best_fold
        Λ_best = _multivariate_indices(
            deepcopy(component_histories[best_fold].terms[best_iteration])
        )

        @info "Selected cross-validation model" component = k fold = best_fold terms = selected_terms[k] validation_objective = mean_test_objectives[best_iteration]

        component = PolynomialMapComponent(Λ_best, rectifier, basis, samples[:, 1:k])
        if reference isa Uniform
            _initialize_uniform_component!(component, samples[:, 1:k], reference)
        end
        map_reference = MapReferenceDensity(reference)
        precomp = PrecomputedBasis(component, samples[:, 1:k])
        res = optimize!(component, map_reference, precomp, optimizer, options)

        train_obj = objective(component, map_reference, precomp) / size(samples, 1)
        @debug "Finished full-data component optimization" component = k objective = train_obj converged = Optim.converged(res) iterations = Optim.iterations(res)

        map_components[k] = component
        fold_histories[k] = component_histories
    end

    # Construct final map from optimized components
    M = PolynomialMap(map_components, reference; forwarddirection = :reference)

    # Return composed map (including linear standardization) and histories
    return ComposedMap(lm, M), fold_histories, selected_terms, selected_fold
end

"""
    optimize_adaptive_transportmapcomponent(train_samples, test_samples, maxterms, reference, rectifier, basis; optimizer, options)

Adaptively optimize a single transport map component by greedily enriching the multi-index set.

# Arguments
- `train_samples::Matrix{Float64}`: Training sample data
- `test_samples::Matrix{Float64}`: Test/validation sample data (can be empty)
- `maxterms::Int`: Maximum number of terms to add
- `reference::Distributions.UnivariateDistribution=Normal()`: Reference distribution
- `rectifier::AbstractRectifierFunction=Softplus()`: Rectifier function to use
- `basis::AbstractPolynomialBasis=_default_basis(reference)`: Polynomial basis

# Keyword Arguments
- `optimizer::Optim.AbstractOptimizer`: Optimization algorithm
- `options::Optim.Options`: Optimizer options

# Returns
- `component::PolynomialMapComponent`: The optimized map component
- `history::OptimizationHistory`: Optimization history for this component
"""
function optimize_adaptive_transportmapcomponent(
        train_samples::Matrix{Float64},
        test_samples::Matrix{Float64},
        maxterms::Int,
        reference::Distributions.UnivariateDistribution = Normal(),
        rectifier::AbstractRectifierFunction = Softplus(),
        basis::AbstractPolynomialBasis = _default_basis(reference);
        optimizer::Optim.AbstractOptimizer = _default_sample_optimizer(MapReferenceDensity(reference)),
        options::Optim.Options = Optim.Options(),
    )
    @assert validate_basis_reference_compatibility(basis, reference) "Invalid combination of reference density and basis."
    initial_terms = reference isa Uniform ? 2 : 1
    @assert maxterms >= initial_terms "maxterms must be at least $initial_terms for $reference"
    map_reference = MapReferenceDensity(reference)
    d = size(train_samples, 2)
    # Initialize multi-index set to contain only zero index (constant term)
    Λ = multivariate_indices(0, d)
    if reference isa Uniform
        push!(Λ, [zeros(Int, d - 1); 1])
    end

    # Initialize history buffer
    history = OptimizationHistory(maxterms - initial_terms + 1)

    # Initialize component with the multi-index set
    @debug "Optimizing initial adaptive term" term = 1 maxterms
    component = PolynomialMapComponent(Λ, rectifier, basis, train_samples)
    if reference isa Uniform
        _initialize_uniform_component!(component, train_samples, reference)
    end

    # Precompute basis for training samples
    train_precomp = PrecomputedBasis(component, train_samples)
    test_precomp = !isempty(test_samples) ? PrecomputedBasis(component, test_samples) : nothing

    res = optimize!(component, map_reference, train_precomp, optimizer, options)

    # Compute and store first iteration
    train_obj = objective(component, map_reference, train_precomp) / size(train_samples, 1)
    test_obj = !isnothing(test_precomp) ? objective(component, map_reference, test_precomp) / size(test_samples, 1) : 0.0
    update_optimization_history!(history, Λ, train_obj, test_obj, Float64[], res, 1)

    # start greedy optimization
    for t in (initial_terms + 1):maxterms
        # Candidates: terms in the reduced margin of Λₜ
        Λ_rm = reduced_margin(Λ)

        # store objective of all candidates and gradients
        gradients = zeros(Float64, length(Λ_rm))
        coeffs = getcoefficients(component)

        # Loop over all candidates in the reduced margin
        for (i, α) in enumerate(Λ_rm)
            Λ_cand = copy(Λ)
            push!(Λ_cand, α)

            # Update component with new multi-index set
            component_cand = PolynomialMapComponent(Λ_cand, rectifier, basis, train_samples)
            # Copy the optimized component coefficients; set other to 0
            setcoefficients!(component_cand, [coeffs..., 0.0]) # set coefficients for existing terms

            # Precompute basis for candidate component
            train_precomp_cand = PrecomputedBasis(component_cand, train_samples)

            # evaluate objective and gradient of objective function
            gradients[i] = abs(objective_gradient!(component_cand, map_reference, train_precomp_cand)[end])
        end

        # select best candidate based on maximum gradient
        α⁺ = Λ_rm[argmax(gradients)]
        push!(Λ, α⁺)

        # Update component with new multi-index set
        @debug "Adding adaptive term" term = t maxterms multiindex = α⁺ gradient = maximum(gradients)
        component = PolynomialMapComponent(Λ, rectifier, basis, train_samples)
        setcoefficients!(component, [coeffs..., 0.0]) # set coefficients for existing terms

        # Recompute precomputed basis with updated component
        train_precomp = PrecomputedBasis(component, train_samples)
        test_precomp = !isempty(test_samples) ? PrecomputedBasis(component, test_samples) : nothing

        if reference isa Uniform
            _ensure_uniform_interior!(component, reference, train_samples, train_precomp)
        end
        res = optimize!(component, map_reference, train_precomp, optimizer, options)

        # Compute and store iteration
        train_obj = objective(component, map_reference, train_precomp) / size(train_samples, 1)
        test_obj = !isnothing(test_precomp) ? objective(component, map_reference, test_precomp) / size(test_samples, 1) : 0.0
        update_optimization_history!(history, Λ, train_obj, test_obj, gradients, res, t - initial_terms + 1)
    end

    return component, history
end

# A constrained fit may end on a support constraint. Move a boundary solution
# slightly toward a feasible affine map before using it to start the next fit.
function _ensure_uniform_interior!(component, reference::Uniform, samples, precomp)
    lower, upper = extrema(reference)
    margin = 0.01 * (upper - lower)
    is_interior() = all(
        value -> lower + margin < value < upper - margin,
        evaluate_M(precomp, component.coefficients, component.rectifier),
    )
    is_interior() && return nothing

    fitted_coefficients = copy(component.coefficients)
    _initialize_uniform_component!(component, samples, reference)
    affine_coefficients = copy(component.coefficients)
    for weight in (0.001, 0.01, 0.1, 1.0)
        component.coefficients .= (1 - weight) .* fitted_coefficients .+ weight .* affine_coefficients
        is_interior() && return nothing
    end
    return nothing
end

# Preserve the former positional interfaces with their standard-normal reference.
function optimize_adaptive_transportmap(
        samples::Matrix{Float64}, maxterms::Vector{Int64}, lm::AbstractLinearMap,
        rectifier::AbstractRectifierFunction,
        basis::AbstractPolynomialBasis = _default_basis(Normal()); kwargs...
    )
    return optimize_adaptive_transportmap(samples, maxterms, lm, Normal(), rectifier, basis; kwargs...)
end

function optimize_adaptive_transportmap(
        samples::Matrix{Float64}, maxterms::Vector{Int64}, k_folds::Int,
        lm::AbstractLinearMap, rectifier::AbstractRectifierFunction,
        basis::AbstractPolynomialBasis = _default_basis(Normal()); kwargs...
    )
    return optimize_adaptive_transportmap(samples, maxterms, k_folds, lm, Normal(), rectifier, basis; kwargs...)
end

function optimize_adaptive_transportmapcomponent(
        train_samples::Matrix{Float64}, test_samples::Matrix{Float64}, maxterms::Int,
        rectifier::AbstractRectifierFunction, basis::AbstractPolynomialBasis,
        optimizer::Optim.AbstractOptimizer, options::Optim.Options,
    )
    return optimize_adaptive_transportmapcomponent(
        train_samples, test_samples, maxterms, Normal(), rectifier, basis; optimizer, options,
    )
end

function _kfold_indices(n_samples::Int, k_folds::Int)
    idx = randperm(n_samples)
    fold_sizes = fill(div(n_samples, k_folds), k_folds)
    remainder = n_samples % k_folds
    for i in 1:remainder
        fold_sizes[i] += 1
    end

    folds = Vector{Vector{Int}}(undef, k_folds)
    pointer = 1
    for fold_id in 1:k_folds
        fold_length = fold_sizes[fold_id]
        stop_idx = pointer + fold_length - 1
        folds[fold_id] = collect(idx[pointer:stop_idx])
        pointer = stop_idx + 1
    end

    return folds
end
