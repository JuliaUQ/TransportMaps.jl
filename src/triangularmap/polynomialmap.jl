"""
    PolynomialMap <: AbstractTriangularMap

Triangular transport map with polynomial basis.

# Fields
- `components::Vector{PolynomialMapComponent{<:AbstractPolynomialBasis}}`: Vector of map components
- `reference::MapReferenceDensity`: Reference density of the map
- `forwarddirection::Symbol`: `:target` for map from density, `:reference` for map from samples

# Constructors
- `PolynomialMap(dimension::Int,degree::Int, referencetype::Symbol=:normal, rectifier::AbstractRectifierFunction=Softplus(), basis::AbstractPolynomialBasis=LinearizedHermiteBasis(), map_type::Symbol=:total)`: Initialize polynomial map for map from density. Reference types: `:normal`, `:uniform` (U(-1,1)), `:uniform01` (U(0,1)).
- `DiagonalMap(dimension::Int, degree::Int, referencetype::Symbol=:normal, rectifier::AbstractRectifierFunction=Softplus(), basis::AbstractPolynomialBasis=LinearizedHermiteBasis())`: Initialize diagonal map with diagonal structure.
- `NoMixedMap(dimension::Int, degree::Int, referencetype::Symbol=:normal, rectifier::AbstractRectifierFunction=Softplus(), basis::AbstractPolynomialBasis=LinearizedHermiteBasis())`: Initialize diagonal map without mixed terms.
"""
mutable struct PolynomialMap <: AbstractTriangularMap
    components::Vector{PolynomialMapComponent{<:AbstractPolynomialBasis}}  # Vector of map components
    reference::MapReferenceDensity
    forwarddirection::Symbol

    function PolynomialMap(
        dimension::Int,
        degree::Int,
        referencetype::Symbol=:normal,
        rectifier::AbstractRectifierFunction=Softplus(),
        basis::AbstractPolynomialBasis=LinearizedHermiteBasis(),
        map_type::Symbol=:total
    )
        @assert referencetype in [:normal, :uniform, :uniform01] "Supported reference types: :normal, :uniform (U(-1,1)), :uniform01 (U(0,1))"

        reference = Dict(
            :normal => Normal(),
            :uniform => Uniform(-1, 1),
            :uniform01 => Uniform(0, 1)
        )

        return PolynomialMap(dimension, degree, reference[referencetype], rectifier, basis, map_type)
    end

    function PolynomialMap(
        dimension::Int,
        degree::Int,
        reference::Distributions.UnivariateDistribution,
        rectifier::AbstractRectifierFunction=Softplus(),
        basis::AbstractPolynomialBasis=LinearizedHermiteBasis(),
        map_type::Symbol=:total
    )
        @assert map_type in [:total, :diagonal, :no_mixed] "Invalid map_type. Supported types are :total, :diagonal, :no_mixed"
        @assert validate_basis_reference_compatibility!(basis, reference) "Invalid combination of reference density and basis. Support of $reference is $(support(reference)), while support of $basis is $(support(basis))."

        components = [PolynomialMapComponent(k, degree, rectifier, basis, reference, map_type) for k in 1:dimension]

        return PolynomialMap(components, reference)
    end

    function PolynomialMap(
        components::Vector{PolynomialMapComponent{T}},
        reference::Distributions.UnivariateDistribution=Normal();
        forwarddirection::Symbol=:target
    ) where T<:AbstractPolynomialBasis
        return new(components, MapReferenceDensity(reference), forwarddirection)
    end
end

# Convenience constructor for DiagonalMap
function DiagonalMap(
    dimension::Int,
    degree::Int,
    reference::Distributions.UnivariateDistribution=Normal(),
    rectifier::AbstractRectifierFunction=Softplus(),
    basis::AbstractPolynomialBasis=LinearizedHermiteBasis()
)
    return PolynomialMap(dimension, degree, reference, rectifier, basis, :diagonal)
end

# Convenience constructor for NoMixedMap
function NoMixedMap(
    dimension::Int,
    degree::Int,
    reference::Distributions.UnivariateDistribution=Normal(),
    rectifier::AbstractRectifierFunction=Softplus(),
    basis::AbstractPolynomialBasis=LinearizedHermiteBasis()
)
    return PolynomialMap(dimension, degree, reference, rectifier, basis, :no_mixed)
end

# Construct PolynomialMap from multi-index sets Λ and given density
function PolynomialMap(
    Λ::Vector{Vector{Vector{Int}}},
    rectifier::AbstractRectifierFunction,
    basis::AbstractPolynomialBasis,
    reference_density::Distributions.UnivariateDistribution=Normal()
)
    d = length(Λ)
    T = typeof(basis)
    components = Vector{PolynomialMapComponent{T}}(undef, d)

    for k in 1:d
        components[k] = PolynomialMapComponent(Λ[k], rectifier, basis, reference_density)
    end

    return PolynomialMap(components; forwarddirection=:target)
end

"""
    evaluate(M::PolynomialMap, z::AbstractVector{<:Real})

Evaluate the polynomial map at point z.

Returns a vector where each component i is Mⁱ(z₁,...,zᵢ).
"""
function evaluate(M::PolynomialMap, z::AbstractVector{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    return [evaluate(component, z[1:i]) for (i, component) in enumerate(M.components)]
end

"""
    evaluate(M::PolynomialMap, Z::AbstractMatrix{<:Real})

Evaluate the polynomial map at multiple points (row-wise) using multithreading.

Each row of Z is a point to evaluate. Returns a matrix where row i contains M(Z[i,:]).
"""
function evaluate(M::PolynomialMap, Z::AbstractMatrix{<:Real})
    @assert size(Z, 2) == length(M.components) "Number of columns must match the dimension of the map"

    n_points = size(Z, 1)
    n_dims = length(M.components)

    # Preallocate result matrix
    results = Matrix{Float64}(undef, n_points, n_dims)

    # Use multithreading to evaluate each point
    Threads.@threads for i in 1:n_points
        z_point = Z[i, :]
        results[i, :] = evaluate(M, z_point)
    end

    return results
end

"""
    gradient_zk(M::PolynomialMap, z::AbstractVector{<:Real})

Compute the diagonal of the Jacobian matrix at point z.

Returns a vector [∂M¹/∂z₁, ∂M²/∂z₂, ..., ∂Mᵈ/∂zᵈ].
"""
function gradient_zk(M::PolynomialMap, z::AbstractVector{<:Real})
    @assert length(z) == length(M.components) "Dimension mismatch: z and components must have same length"
    # Compute the derivatives ∂Mᵏ/∂zₖ for each component
    return [partial_derivative_zk(component, z[1:i]) for (i, component) in enumerate(M.components)]
end


"""
    gradient_zk(M::PolynomialMap, Z::AbstractMatrix{<:Real})

Compute the diagonal Jacobian elements at multiple points using multithreading.

Each row of Z is a point. Returns a matrix where row i contains the diagonal of J_M(Z[i,:]).
"""
function gradient_zk(M::PolynomialMap, Z::AbstractMatrix{<:Real})
    @assert size(Z, 2) == length(M.components) "Number of columns must match the dimension of the map"

    n_points = size(Z, 1)
    n_dims = length(M.components)

    # Preallocate result matrix
    results = Matrix{Float64}(undef, n_points, n_dims)

    # Use multithreading to compute gradient for each point
    Threads.@threads for i in 1:n_points
        z_point = Z[i, :]
        results[i, :] = gradient_zk(M, z_point)
    end

    return results
end

# Gradient of the map with respect to the coefficients at z
"""
    gradient_coefficients(M::PolynomialMap, z::AbstractVector{<:Real})

Compute the gradient of the polynomial map with respect to all its coefficients at point z.

For a triangular polynomial map M(z) = [M¹(z₁), M²(z₁,z₂), ..., Mᵈ(z₁,...,zᵈ)],
this function returns the gradient matrix where:
- Each row i corresponds to component Mⁱ
- Each column j corresponds to a coefficient across all components

The coefficients are ordered by component: [c₁₁, c₁₂, ..., c₂₁, c₂₂, ..., cᵈₙ]
where cᵢⱼ is the j-th coefficient of the i-th component.

# Arguments
- `M::PolynomialMap`: The polynomial map
- `z::Vector{Float64}`: Point at which to evaluate the gradient

# Returns
- `Matrix{Float64}`: Gradient matrix of size (dimension × total_coefficients)
  - Element (i,j) = ∂Mⁱ/∂cⱼ at point z

# Examples
```julia
# Create a 2D polynomial map
M = PolynomialMap(2, 2, Softplus())
setcoefficients!(M, randn(numbercoefficients(M)))

# Evaluate gradient at point z = [0.5, 1.2]
z = [0.5, 1.2]
grad_matrix = gradient_coefficients(M, z)

# grad_matrix[i, j] = ∂Mⁱ/∂cⱼ
```
"""
function gradient_coefficients(M::PolynomialMap, z::AbstractVector{<:Real})
    @assert length(z) == length(M.components) "Dimension mismatch: z and components must have same length"

    n_dimensions = length(M.components)
    n_total_coeffs = numbercoefficients(M)

    # Initialize gradient matrix: (n_dimensions × n_total_coeffs)
    gradient_matrix = zeros(Float64, n_dimensions, n_total_coeffs)

    # Track coefficient index across all components
    coeff_offset = 1

    for (i, component) in enumerate(M.components)
        n_component_coeffs = length(component.coefficients)

        # For component i, compute gradient with respect to its own coefficients
        # This gives ∂Mⁱ/∂cᵢⱼ for all j in component i
        component_grad = gradient_coefficients(component, z[1:i])

        # Fill in the gradient matrix
        gradient_matrix[i, coeff_offset:coeff_offset+n_component_coeffs-1] = component_grad

        # All other coefficients don't affect this component (triangular structure)
        # So gradient_matrix[i, other_indices] remains zero

        coeff_offset += n_component_coeffs
    end

    return gradient_matrix
end

"""
    gradient_coefficients(M::PolynomialMap, Z::AbstractMatrix{<:Real})

Compute coefficient gradients at multiple points using multithreading.

Returns a 3D array of size (n_points × n_dims × n_coeffs).
"""
function gradient_coefficients(M::PolynomialMap, Z::AbstractMatrix{<:Real})
    @assert size(Z, 2) == length(M.components) "Number of columns must match the dimension of the map"

    n_points = size(Z, 1)
    n_dims = length(M.components)
    n_coeffs = numbercoefficients(M)

    # Preallocate result: 3D array (n_points × n_dims × n_coeffs)
    results = Array{Float64,3}(undef, n_points, n_dims, n_coeffs)

    # Use multithreading to compute gradient for each point
    Threads.@threads for i in 1:n_points
        z_point = Z[i, :]
        results[i, :, :] = gradient_coefficients(M, z_point)
    end

    return results
end

"""
    jacobian(M::PolynomialMap, z::AbstractVector{<:Real})

Compute the Jacobian determinant det(J_M(z)) at point z.

For triangular maps, this is the product of diagonal elements: ∏ᵢ ∂Mⁱ/∂zᵢ.
"""
function jacobian(M::PolynomialMap, z::AbstractVector{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"
    # Compute the jacobian determinant as ∏ₖ ∂Mᵏ/∂zₖ
    return prod(gradient_zk(M, z))
end

"""
    jacobian(M::PolynomialMap, Z::AbstractMatrix{<:Real})

Compute Jacobian determinants at multiple points using multithreading.

Returns a vector where element i is det(J_M(Z[i,:])).
"""
function jacobian(M::PolynomialMap, Z::AbstractMatrix{<:Real})
    @assert size(Z, 2) == length(M.components) "Number of columns must match the dimension of the map"

    n_points = size(Z, 1)

    # Preallocate result vector
    results = Vector{Float64}(undef, n_points)

    # Use multithreading to compute Jacobian for each point
    Threads.@threads for i in 1:n_points
        z_point = Z[i, :]
        results[i] = jacobian(M, z_point)
    end

    return results
end


"""
    jacobian_logdet_gradient(M::PolynomialMap, z::AbstractVector{<:Real})

Compute the gradient of log|det J_M(z)| with respect to all coefficients.

For triangular maps: ∂log|det J_M|/∂c = ∑ᵢ (1/(∂Mⁱ/∂zᵢ)) * ∂²Mⁱ/(∂zᵢ∂c).
"""
function jacobian_logdet_gradient(M::PolynomialMap, z::AbstractVector{<:Real})
    n_coeffs = numbercoefficients(M)
    gradient = zeros(Float64, n_coeffs)

    # For triangular maps: log|det J_M| = Σᵢ log(∂Mᵢ/∂zᵢ)
    # So ∂log|det J_M|/∂c = Σᵢ (1/(∂Mᵢ/∂zᵢ)) * ∂²Mᵢ/(∂zᵢ∂c)

    coeff_offset = 1
    for (i, component) in enumerate(M.components)
        n_comp_coeffs = length(component.coefficients)

        # Compute ∂Mᵢ/∂zᵢ (diagonal element of Jacobian)
        diagonal_deriv = partial_derivative_zk(component, z[1:i])

        # Compute gradient of ∂Mᵢ/∂zᵢ with respect to component's coefficients
        # This requires the gradient of partial_derivative_xk w.r.t. coefficients
        comp_grad = partial_derivative_zk_gradient_coefficients(component, z[1:i])

        # Add contribution: (1/diagonal_deriv) * comp_grad
        coeff_range = coeff_offset:coeff_offset+n_comp_coeffs-1
        gradient[coeff_range] += comp_grad / diagonal_deriv

        coeff_offset += n_comp_coeffs
    end

    return gradient
end

"""
    jacobian_logdet_gradient(M::PolynomialMap, Z::AbstractMatrix{<:Real})

Compute log-determinant gradients at multiple points using multithreading.

Returns a matrix of size (n_points × n_coeffs).
"""
function jacobian_logdet_gradient(M::PolynomialMap, Z::AbstractMatrix{<:Real})
    @assert size(Z, 2) == length(M.components) "Number of columns must match the dimension of the map"

    n_points = size(Z, 1)
    n_coeffs = numbercoefficients(M)

    results = Matrix{Float64}(undef, n_points, n_coeffs)

    Threads.@threads for i in 1:n_points
        z_point = Z[i, :]
        results[i, :] = jacobian_logdet_gradient(M, z_point)
    end

    return results
end

"""
    inverse(M::PolynomialMap, x::AbstractVector{<:Real}, k::Int=numberdimensions(M))

Compute the inverse of the first k components of the map at point x.

Returns z such that M(z)[1:k] = x[1:k].
"""
function inverse(M::PolynomialMap, x::AbstractVector{<:Real}, k::Int=numberdimensions(M))
    @assert k <= length(x) <= numberdimensions(M) "x must have at least k dimensions and at most the map dimension"

    # Initialize the inverse map
    z = Vector{Float64}(undef, k)
    for (i, component) in enumerate(M.components[1:k])
        z[i] = inverse(component, z[1:i-1], x[i])
    end

    return z
end

"""
    inverse(M::PolynomialMap, X::AbstractMatrix{<:Real}, k::Int=numberdimensions(M))

Compute the inverse at multiple points using multithreading.

Returns a matrix where row i contains M⁻¹(X[i,:])[1:k].
"""
function inverse(M::PolynomialMap, X::AbstractMatrix{<:Real}, k::Int=numberdimensions(M))
    @assert k <= size(X, 2) == numberdimensions(M) "X must have at least k columns and at most the map dimension"

    n_points = size(X, 1)

    # Preallocate result matrix
    results = Matrix{Float64}(undef, n_points, k)

    # Use multithreading to compute inverse for each point
    Threads.@threads for i in 1:n_points
        x_point = X[i, :]
        results[i, :] = inverse(M, x_point, k)
    end

    return results
end

"""
    inverse_jacobian(M::PolynomialMap, x::AbstractVector{<:Real})

Compute the Jacobian determinant of the inverse map at point x.

Returns det(J_{M⁻¹}(x)) = 1/det(J_M(M⁻¹(x))).
"""
function inverse_jacobian(M::PolynomialMap, x::AbstractVector{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"

    # For triangular maps, det(J_{M⁻¹}(x)) = 1/det(J_M(M⁻¹(x)))
    # where J_M is the Jacobian of the forward map M
    return 1.0 / jacobian(M, inverse(M, x))
end

"""
    inverse_jacobian(M::PolynomialMap, X::AbstractMatrix{<:Real})

Compute inverse Jacobian determinants at multiple points using multithreading.

Returns a vector where element i is det(J_{M⁻¹}(X[i,:])).
"""
function inverse_jacobian(M::PolynomialMap, X::AbstractMatrix{<:Real})
    @assert size(X, 2) == length(M.components) "Number of columns must match the dimension of the map"

    n_points = size(X, 1)

    # Preallocate result vector
    results = Vector{Float64}(undef, n_points)

    # Use multithreading to compute inverse Jacobian for each point
    Threads.@threads for i in 1:n_points
        x_point = X[i, :]
        results[i] = inverse_jacobian(M, x_point)
    end

    return results
end

"""
    pullback(M::PolynomialMap, x::AbstractVector{<:Real})

Compute the pullback density π̂(x) = ρ(M⁻¹(x)) |det J_{M⁻¹}(x)|.

Maps the reference density through M to approximate the target density.
"""
function pullback(M::PolynomialMap, x::AbstractVector{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"

    value = M.forwarddirection == :target ? pdf(M.reference, inverse(M, x)) * abs(inverse_jacobian(M, x)) :
            pdf(M.reference, evaluate(M, x)) * abs(jacobian(M, x))

    # Compute pull-back density π̂(x) = ρ(M⁻¹(x)) * |det J(M^-1(x))|
    return value
end

"""
    pullback(M::PolynomialMap, X::AbstractMatrix{<:Real})

Compute pullback densities at multiple points using multithreading.

Returns a vector where element i is the pullback density at X[i,:].
"""
function pullback(M::PolynomialMap, X::AbstractMatrix{<:Real})
    @assert size(X, 2) == length(M.components) "Number of columns must match the dimension of the map"

    n_points = size(X, 1)

    # Preallocate result vector
    results = Vector{Float64}(undef, n_points)

    # Use multithreading to compute pullback for each point
    Threads.@threads for i in 1:n_points
        x_point = X[i, :]
        results[i] = pullback(M, x_point)
    end

    return results
end


"""
    pushforward(M::PolynomialMap, target::MapTargetDensity, z::AbstractVector{<:Real})

Compute the pushforward density ρ(z) = π(M(z)) |det J_M(z)|.

Maps the target density back through M to the reference space.
"""
function pushforward(M::PolynomialMap, target::MapTargetDensity, z::AbstractVector{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    value = M.forwarddirection == :target ? pdf(target, evaluate(M, z)) * abs(jacobian(M, z)) :
            error("Can't evaluate pushforward for a map from samples!")

    # Compute push-forward density ρ(z) = π(M(z)) * |det J(M(z))|
    return value
end

"""
    pushforward(M::PolynomialMap, target::MapTargetDensity, Z::AbstractMatrix{<:Real})

Compute pushforward densities at multiple points using multithreading.

Returns a vector where element i is the pushforward density at Z[i,:].
"""
function pushforward(M::PolynomialMap, target::MapTargetDensity, Z::AbstractMatrix{<:Real})
    @assert size(Z, 2) == length(M.components) "Number of columns must match the dimension of the map"

    n_points = size(Z, 1)

    # Preallocate result vector
    results = Vector{Float64}(undef, n_points)

    # Use multithreading to compute pushforward for each point
    Threads.@threads for i in 1:n_points
        z_point = Z[i, :]
        results[i] = pushforward(M, target, z_point)
    end

    return results
end

"""
    setcoefficients!(M::PolynomialMap, coefficients::AbstractVector{<:Real})

Set all coefficients of the polynomial map.

Coefficients are ordered by component: [c₁₁, c₁₂, ..., c₂₁, c₂₂, ..., cᵈₙ].
"""
function setcoefficients!(M::PolynomialMap, coefficients::AbstractVector{<:Real})
    counter = 1
    for component in M.components
        setcoefficients!(component, coefficients[counter:counter+length(component.basisfunctions)-1])
        counter += length(component.basisfunctions)
    end
end

"""
    getcoefficients(M::PolynomialMap)

Get all coefficients from the polynomial map.

Returns a vector with coefficients ordered by component.
"""
function getcoefficients(M::PolynomialMap)
    coefficients = Vector{Float64}(undef, numbercoefficients(M))
    counter = 1
    for component in M.components
        coefficients[counter:counter+length(component.basisfunctions)-1] .= getcoefficients(component)
        counter += length(component.basisfunctions)
    end
    return coefficients
end

"""
    setforwarddirection!(M::PolynomialMap, forwarddirection::Symbol)

Set the forward direction of the map.

Valid directions: `:target` (map from density) or `:reference` (map from samples).
"""
function setforwarddirection!(M::PolynomialMap, forwarddirection::Symbol)
    @assert forwarddirection in [:reference, :target] "Direction must be :reference, :target"
    M.forwarddirection = forwarddirection
end

"""
    numbercoefficients(M::PolynomialMap)

Return the total number of coefficients in the polynomial map.
"""
function numbercoefficients(M::PolynomialMap)
    if numberdimensions(M) == 0
        return 0
    else
        return sum(length(component.coefficients) for component in M.components)
    end
end

"""
    numberdimensions(M::PolynomialMap)

Return the number of dimensions (components) of the polynomial map.
"""
function numberdimensions(M::PolynomialMap)
    return length(M.components)
end

"""
    initializemapfromsamples!(M::PolynomialMap, samples::Matrix{Float64})

Initialize the map for optimization from samples.

Sets the forward direction to `:reference` and prepares components for sample-based optimization.
"""
function initializemapfromsamples!(M::PolynomialMap, samples::Matrix{Float64})
    # Check dimensions
    @assert size(samples, 2) == numberdimensions(M) "Samples must have the same number of columns as number of map components in M"
    setforwarddirection!(M, :reference)
    new_components = Vector{PolynomialMapComponent}(undef, length(M.components))

    # save coefficients
    map_coefficients = getcoefficients(M)

    for (i, component) in enumerate(M.components)

        k = component.index
        d = degree(component)
        rec = component.rectifier
        basis = component.basisfunctions[1].univariatebases[1]

        new_components[i] = PolynomialMapComponent(k, d, rec, basis, samples)
    end

    # Assign components
    M.components .= new_components

    # re-set coefficients
    setcoefficients!(M, map_coefficients)
end

"""
    length(M::PolynomialMap)

Return the number of components (dimension) of the map.
"""
Base.length(M::PolynomialMap) = length(M.components)

"""
    getindex(M::PolynomialMap, i::Int)

Get the i-th component of the polynomial map. Allows using M[i] syntax.
"""
Base.@propagate_inbounds Base.getindex(M::PolynomialMap, i::Int) = getindex(M.components, i)

"""
    (M::PolynomialMap)(z)

Make PolynomialMap callable. Equivalent to `evaluate(M, z)`.
"""
Base.@propagate_inbounds (M::PolynomialMap)(z::AbstractArray{<:Real}) = evaluate(M, z)

# Display method for PolynomialMap
function Base.show(io::IO, M::PolynomialMap)
    n_dims = length(M.components)
    n_coeffs = numbercoefficients(M)

    # Get common properties from the first component
    if n_dims > 0
        first_component = M.components[1]
        max_degree = maximum(sum(basis.multiindexset) for basis in first_component.basisfunctions)

        # Get basis type
        basis_name = string(basistype(first_component.basisfunctions[1]))

        # Get rectifier type
        rectifier_type = typeof(first_component.rectifier)
        rectifier_name = string(rectifier_type)

        print(io, "PolynomialMap(")
        print(io, "$n_dims-dimensional, ")
        print(io, "degree=$max_degree, ")
        print(io, "basis=$basis_name, ")
        print(io, "reference=$(M.reference.densitytype), ")
        print(io, "rectifier=$rectifier_name, ")
        print(io, "$n_coeffs total coefficients)")
    else
        print(io, "PolynomialMap(empty)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", M::PolynomialMap)
    n_dims = length(M.components)
    n_coeffs = numbercoefficients(M)

    println(io, "PolynomialMap:")
    println(io, "  Dimensions: $n_dims")
    println(io, "  Total coefficients: $n_coeffs")
    println(io, "  Reference density: $(M.reference.densitytype)")

    if n_dims > 0
        # Get properties from the first component (assuming all components have similar properties)
        first_component = M.components[1]
        max_degree = maximum(sum(basis.multiindexset) for basis in first_component.basisfunctions)

        # Get basis type
        basis_name = basis_name = string(basistype(first_component.basisfunctions[1]))

        # Get rectifier type
        rectifier_type = typeof(first_component.rectifier)
        rectifier_name = string(rectifier_type)

        println(io, "  Maximum degree: $max_degree")
        println(io, "  Basis: $basis_name")
        println(io, "  Rectifier: $rectifier_name")

        # Show components summary
        println(io, "  Components:")
        for (i, component) in enumerate(M.components)
            n_basis = length(component.basisfunctions)
            println(io, "    Component $i: $n_basis basis functions")
        end

        # Show coefficient statistics if they're set
        all_coeffs = getcoefficients(M)
        if all(isfinite, all_coeffs)
            coeff_min = minimum(all_coeffs)
            coeff_max = maximum(all_coeffs)
            coeff_mean = sum(all_coeffs) / length(all_coeffs)
            println(io, "  Coefficients: min=$coeff_min, max=$coeff_max, mean=$coeff_mean")
        else
            println(io, "  Coefficients: uninitialized")
        end
    else
        println(io, "  (Empty map)")
    end
end

# Check compatibility of reference densities and selected bassis
function validate_basis_reference_compatibility!(basis::AbstractPolynomialBasis, reference::Distributions.UnivariateDistribution)
    # Check Hermite basis family with Gaussian reference
    if basis isa Union{HermiteBasis,LinearizedHermiteBasis,CubicSplineHermiteBasis,GaussianWeightedHermiteBasis}
        if !(reference isa Normal)
            return false
        end
    end

    # Check Legendre basis with Uniform(-1, 1)
    if basis isa LegendreBasis
        if reference isa Uniform
            if !(reference.a ≈ -1.0 && reference.b ≈ 1.0)
                return false
            end
        elseif !(reference isa Uniform)
            return false
        end
    end

    # Check Shifted Legendre basis with Uniform(0, 1)
    if basis isa ShiftedLegendreBasis
        if reference isa Uniform
            if !(reference.a ≈ 0.0 && reference.b ≈ 1.0)
                return false
            end
        elseif !(reference isa Uniform)
            return false
        end
    end

    return true
end
