"""
    optimize_adaptive_transportmap(
        target::AbstractMapDensity,
        quadrature::AbstractQuadratureWeights,
        maxterms::Int;
        kwargs...
    )

Adaptively optimize a triangular transport map from a target density by greedily enriching
the multi-index set across all components simultaneously.

# Arguments
- `target::AbstractMapDensity`: Target density to approximate
- `quadrature::AbstractQuadratureWeights`: Quadrature points and weights for integration
- `maxterms::Int`: Maximum total number of terms to add across all components

# Keyword Arguments
- `initial_map::Union{Nothing,PolynomialMap}=nothing`: Initial transport map structure
- `rectifier::AbstractRectifierFunction=Softplus()`: Rectifier function to use
- `basis::AbstractPolynomialBasis=LinearizedHermiteBasis()`: Polynomial basis
- `optimizer::Optim.AbstractOptimizer=LBFGS()`: Optimization algorithm
- `options::Optim.Options=Optim.Options()`: Optimizer options
- `λ1::Real=0`: Strength ``\\lambda_1`` of the smoothed L1 penalty
- `λ2::Real=0`: Strength ``\\lambda_2`` of the L2 penalty
- `l1_eps::Real=1e-8`: Positive smoothing parameter ``\\varepsilon`` for the L1 penalty
- `δ::Real=1e-9`: Stability perturbation ``\\delta`` for mapped quadrature points during target
  density evaluation
- `interactions_only::Bool=false`: Penalize only terms involving multiple coordinates
- `validation::Union{AbstractQuadratureWeights,Nothing}=nothing`: Quadrature rule used for validation diagnostics

# Returns
- `M::PolynomialMap`: The optimized triangular transport map (with best validation variance diagnostic)
- `history::OptimizationHistory`: History of optimization iterations
"""
function optimize_adaptive_transportmap(
        target::AbstractMapDensity,
        quadrature::AbstractQuadratureWeights,
        maxterms::Int;
        initial_map::Union{Nothing, PolynomialMap} = nothing,
        rectifier::AbstractRectifierFunction = Softplus(),
        basis::AbstractPolynomialBasis = LinearizedHermiteBasis(),
        reference_density::Distributions.UnivariateDistribution = Normal(),
        optimizer::Optim.AbstractOptimizer = LBFGS(),
        options::Optim.Options = Optim.Options(),
        λ1::Real = 0.0,
        λ2::Real = 0.0,
        l1_eps::Real = 1.0e-8,
        δ::Real = 1.0e-9,
        interactions_only::Bool = false,
        validation::Union{AbstractQuadratureWeights, Nothing} = nothing
    )
    d = size(quadrature.points, 2)

    if isnothing(initial_map)
        # Initialize map with constant terms only
        Λ = [multivariate_indices(0, k) for k in 1:d]
        M = PolynomialMap(Λ, rectifier, basis, reference_density)
    else
        @assert numbercoefficients(initial_map) <= maxterms "Initial map has more coefficients than maxterms=$maxterms"
        M = deepcopy(initial_map)
        setforwarddirection!(M, :target)
    end

    num_initial_coefficients = numbercoefficients(M)
    @debug "Initialized adaptive map" initial_coefficients = num_initial_coefficients maxterms

    # Initialize history tracking
    history = MapOptimizationResult(maxterms - num_initial_coefficients + 1)

    # Precompute basis for quadrature points
    precomp = PrecomputedMapBasis(M, quadrature.points, quadrature.weights)

    # Optimize initial map
    res = optimize!(
        M, target, precomp;
        optimizer, options, δ, λ1, λ2, l1_eps, interactions_only,
    )
    train_obj = Optim.minimum(res)

    @debug "Initial training KL divergence" objective = train_obj

    # Perform validation if not set to nothing
    if !isnothing(validation)
        validation_obj = kldivergence(M, target, validation; δ)
        @debug "Initial validation KL divergence" objective = validation_obj
    else
        validation_obj = NaN
    end

    update_optimization_history!(
        history,
        deepcopy(M),
        train_obj,
        validation_obj,
        Float64[],
        res,
        1,
    )

    # Greedy optimization loop
    for iteration in (num_initial_coefficients + 1):maxterms
        @debug "Starting adaptive term selection" iteration maxterms

        # Collect all candidate terms from reduced margins of all components
        candidates = Vector{Tuple{Int, Vector{Int}}}()  # (component_idx, multi_index)

        for k in 1:d
            Λᵣₘᵏ = reduced_margin(getmultivariateindices(M[k]))
            for α in Λᵣₘᵏ
                push!(candidates, (k, α))
            end
        end

        @debug "Evaluating candidate terms" iteration candidates = length(candidates)

        # Evaluate all candidates by computing gradient magnitude of KL divergence
        gradient_metrics = zeros(Float64, length(candidates))

        for (i, (k, α)) in enumerate(candidates)
            # Construct candidate map
            M_cand = deepcopy(M)
            update_multiindexset!(M_cand, α, k)

            # Compute gradient of KL divergence
            precomp_cand = PrecomputedMapBasis(M_cand, quadrature.points, quadrature.weights)
            grad = kldivergence_gradient(M_cand, target, precomp_cand; δ)

            # Get gradient component corresponding to the new coefficient (last one for component k)
            # Find position of new coefficient in the full coefficient vector
            coeff_offset = k == 1 ? 0 : sum(numbercoefficients(M_cand[j]) for j in 1:(k - 1))
            new_coeff_idx = coeff_offset + numbercoefficients(M_cand[k])

            # Use absolute value of gradient as metric
            gradient_metrics[i] = abs(grad[new_coeff_idx])
        end

        # Try candidates in descending order of gradient magnitude and keep the first converged one
        sorted_candidate_indices = sortperm(gradient_metrics, rev = true)
        candidate_selected = false

        for cand_idx in sorted_candidate_indices
            k_cand, α_cand = candidates[cand_idx]

            @debug "Trying candidate term" iteration component = k_cand multiindex = α_cand gradient = gradient_metrics[cand_idx]

            M_trial = deepcopy(M)
            update_multiindexset!(M_trial, α_cand, k_cand)

            precomp_trial = PrecomputedMapBasis(M_trial, quadrature.points, quadrature.weights)
            res_trial = optimize!(
                M_trial, target, precomp_trial;
                optimizer, options, δ, λ1, λ2, l1_eps, interactions_only,
            )

            if Optim.converged(res_trial)
                M = M_trial
                precomp = precomp_trial
                res = res_trial
                candidate_selected = true
                break
            else
                @debug "Candidate did not converge" iteration component = k_cand multiindex = α_cand
            end
        end

        if candidate_selected
            # Compute objectives for accepted candidate
            train_obj = Optim.minimum(res)
            @debug "Accepted adaptive term" iteration training_objective = train_obj

            if !isnothing(validation)
                validation_obj = kldivergence(M, target, validation; δ)
                @debug "Validation KL divergence" iteration objective = validation_obj
            else
                validation_obj = NaN
            end
        else
            @warn "No converged candidate found; map remains unchanged" iteration
            # Keep the previous objective values and optimization result in history.
        end

        # Store in history
        iter_idx = iteration - num_initial_coefficients + 1
        update_optimization_history!(
            history,
            deepcopy(M),
            train_obj,
            validation_obj,
            gradient_metrics,
            res,
            iter_idx,
        )

    end

    # Select model with best KL divergence
    if !isnothing(validation)
        best_iteration = argmin(history.test_objectives)
        @info "Selected best adaptive map" iteration = best_iteration training_objective = history.train_objectives[best_iteration] validation_objective = history.test_objectives[best_iteration]
    else
        best_iteration = argmin(history.train_objectives)
        @info "Selected best adaptive map" iteration = best_iteration training_objective = history.train_objectives[best_iteration]
    end

    # Get best map
    M_best = history.maps[best_iteration]

    return M_best, history
end

# Update the polynomial map with a new multi-index α in component k
function update_multiindexset!(
        M::PolynomialMap,
        α::Vector{Int},
        k::Int,
    )
    # Get k-th component to update
    component = M[k]
    coeffs = getcoefficients(component)

    # Get the current multi-index set for the k-th component and add new index
    Λ = getmultivariateindices(component)
    push!(Λ, α)

    # Reconstruct map component with updated multi-index set
    M.components[k] = PolynomialMapComponent(Λ, component.rectifier, getbasis(component), M.reference.densitytype)
    setcoefficients!(M.components[k], [coeffs..., 0.0])  # Initialize new coefficient to zero
    return nothing

end
