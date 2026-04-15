# Tensor grid (full grid) and Smolyak grid

# Multi-Index set
function generate_multi_indices(d::Int, target_sum::Int)
    if d == 1
        return [[target_sum]]
    end

    indices = Vector{Vector{Int}}()
    for i1 in 0:target_sum
        append!(indices, ([i1; sub_idx] for sub_idx in generate_multi_indices(d - 1, target_sum - i1)))
    end

    return indices
end

function create_tensor_product(nodes_sets::Vector{Vector{Float64}}, weights_sets::Vector{Vector{Float64}})
    d = length(nodes_sets)
    ranges = [eachindex(nodes_sets[i]) for i in 1:d]
    entries = [
        (
            [nodes_sets[i][idx[i]] for i in 1:d],
            prod(weights_sets[i][idx[i]] for i in 1:d),
        ) for idx in Iterators.product(ranges...)
    ]

    tensor_nodes = first.(entries)
    tensor_weights = last.(entries)
    return tensor_nodes, tensor_weights
end

# Combine duplicate nodes to get reduced sparse grid
function combine_duplicate_nodes(
    nodes::Vector{Vector{Float64}}, weights::Vector{Float64}, tol::Float64=1e-14
)
    unique_nodes = Vector{Vector{Float64}}()
    combined_weights = Float64[]

    for (node, weight) in zip(nodes, weights)
        idx = findfirst(existing_node -> all(abs.(node .- existing_node) .< tol), unique_nodes)
        if isnothing(idx)
            push!(unique_nodes, copy(node))
            push!(combined_weights, weight)
        else
            combined_weights[idx] += weight
        end
    end

    return unique_nodes, combined_weights
end

# Generate (sparse) Smolyak grid
function smolyak_points(
    d::Int, level::Int, basis::AbstractQuadratureKnots, sparse::Bool=true; tol::Float64=1e-14
)
    all_nodes = Vector{Vector{Float64}}()
    all_weights = Float64[]

    for total_level in max(0, level - d + 1):level
        multi_indices = generate_multi_indices(d, total_level)
        coeff = (-1)^(level - total_level) * binomial(d - 1, level - total_level)

        for mi in multi_indices
            rules_1d = [basis(mi[j]) for j in 1:d]
            nodes_1d = first.(rules_1d)
            weights_1d = last.(rules_1d)

            tensor_nodes, tensor_weights = create_tensor_product(nodes_1d, weights_1d)
            append!(all_nodes, tensor_nodes)
            append!(all_weights, coeff .* tensor_weights)
        end
    end

    # Combine duplicate points
    if sparse
        unique_nodes, combined_weights = combine_duplicate_nodes(all_nodes, all_weights, tol)
    else
        unique_nodes, combined_weights = all_nodes, all_weights
    end

    # Only keep nodes with weight greater tolerance
    keep = findall(w -> abs(w) > tol, combined_weights)
    if isempty(keep)
        return zeros(Float64, 0, d), Float64[]
    end

    filtered_weights = combined_weights[keep]
    points = reduce(vcat, (reshape(unique_nodes[i], 1, d) for i in keep))

    return points, filtered_weights
end

# Generate full-order tensor-product grid at a fixed 1D level
function full_tensor_points(
    d::Int, level::Int, basis::AbstractQuadratureKnots; tol::Float64=eps(Float64)
)
    rules_1d = [basis(level) for _ in 1:d]
    nodes_1d = first.(rules_1d)
    weights_1d = last.(rules_1d)

    nodes, weights = create_tensor_product(nodes_1d, weights_1d)

    keep = findall(w -> abs(w) > tol, weights)
    if isempty(keep)
        return zeros(Float64, 0, d), Float64[]
    end

    filtered_weights = weights[keep]
    points = reduce(vcat, (reshape(nodes[i], 1, d) for i in keep))

    return points, filtered_weights
end
