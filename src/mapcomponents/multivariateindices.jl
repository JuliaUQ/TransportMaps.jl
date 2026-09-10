# Multivariate index set generation and reduced margin of multi-index sets

"""
    multivariate_indices(p::Int, k::Int; mode::Symbol = :total, q::Real = 1.0)

Generate a multi-index set ``\\Lambda\\subset\\mathbb{N}_0^k`` for a multivariate
polynomial basis. The supported truncations are

```math
\\begin{aligned}
\\Lambda_{\\mathrm{total}}
  &= \\{\\alpha:\\|\\alpha\\|_1\\leq p\\},\\\\
\\Lambda_{\\mathrm{diagonal}}
  &= \\{\\alpha:\\|\\alpha\\|_1\\leq p,\\ \\alpha_j=0\\text{ for }j\\neq k\\},\\\\
\\Lambda_{\\mathrm{no\\ mixed}}
  &= \\{\\alpha:\\|\\alpha\\|_1\\leq p,\\ \\|\\alpha\\|_0\\leq1\\},\\\\
\\Lambda_{\\mathrm{hyperbolic}}
  &= \\{\\alpha:\\|\\alpha\\|_q\\leq p\\}.
\\end{aligned}
```

# Arguments
- `p::Int`: Maximum total degree of the polynomials.
- `k::Int`: Dimension of the input space.
- `mode::Symbol`: Type of multi-index set to generate. Supported modes are:
    - `:total`: Total-order multi-indices (default).
    - `:diagonal`: Diagonal multi-indices for a fixed coordinate k.
    - `:no_mixed`: No-mixed multi-indices.
    - `:hyperbolic`: Hyperbolic truncation scheme by `q`-norm.
- `q::Real`: Quasi-norm parameter for hyperbolic truncation.
  The choice ``q=1`` recovers total-order truncation.

# Returns
- `Vector{Vector{Int}}`: A vector of multi-indices, where each multi-index is represented as a vector of integers.

# References
[marzouk2016](@cite), [blatman2011](@cite)

"""
function multivariate_indices(p::Int, k::Int; mode::Symbol = :total, q::Real = 1.0)
    @assert p >= 0 "Degree p must be non-negative"
    @assert k >= 1 "Dimension k must be at least 1"
    @assert mode in [:total, :diagonal, :no_mixed, :hyperbolic] "Unknown mode: $mode. Supported modes are :total, :diagonal, :no_mixed, :hyperbolic"

    # Special-case p == 0: only the constant multi-index of zeros
    if p == 0
        return [zeros(Int, k)]
    end

    if mode == :total
        # total-order multi-indices
        # A_k^{TO} = { α : ||α||_1 <= p and α_i = 0 for all i > k}
        return total_order_indices(p, k)

    elseif mode == :diagonal
        # Diagonal multi-index set for a fixed coordinate k:
        # A_k^D = { α : ||α||_1 <= p  and α_i = 0 for all i != k }
        # i.e., vectors with only the k-th entry possibly non-zero (0..p)
        # diagonal for the fixed component k: only α_{k} may be non-zero
        inds = Vector{Vector{Int}}()
        for t in 0:p
            v = zeros(Int, k)
            v[k] = t
            push!(inds, v)
        end
        return inds

    elseif mode == :no_mixed
        # No-mixed terms (possibly restricted to first k coordinates):
        # A_k^{NM} = { α : ||α||_1 <= p, α_i * α_l = 0 for i != l, and α_i = 0 for i > k }
        inds = Vector{Vector{Int}}()
        # constant term
        push!(inds, zeros(Int, k))
        # for each allowed coordinate 1..k, add pure powers 1..p
        for j in 1:k
            for t in 1:p
                v = zeros(Int, k)
                v[j] = t
                push!(inds, v)
            end
        end
        return inds

    elseif mode == :hyperbolic
        # Hyperbolic truncation:
        # A_k^{q} = { α : ||α||_q <= p and α_i = 0 for all i > k}
        return hyperbolic_truncation(p, k, q)
    end

end

function total_order_indices(p::Int, k::Int)
    inds = Vector{Vector{Int}}()
    current = zeros(Int, k)
    _fill_total_order_product_order!(inds, current, k, 0, p)
    return inds
end

function _fill_total_order_product_order!(
        inds::Vector{Vector{Int}},
        current::Vector{Int},
        pos::Int,
        partial_sum::Int,
        p::Int
    )
    remaining = p - partial_sum
    remaining < 0 && return

    if pos == 1
        # First coordinate is fastest, matching Iterators.product ordering.
        for a1 in 0:remaining
            current[1] = a1
            push!(inds, copy(current))
        end
        return nothing
    end

    # Higher coordinates are outer loops.
    for a in 0:remaining
        current[pos] = a
        _fill_total_order_product_order!(inds, current, pos - 1, partial_sum + a, p)
    end
    return nothing
end

function hyperbolic_truncation(p::Int, k::Int, q::Real)
    @assert 0 < q <= 1 "q must be in (0,1]"
    if q == 1
        return total_order_indices(p, k)
    end

    inds = Vector{Vector{Int}}()
    for α in total_order_indices(p, k)
        if q_norm(α, q) <= p + 1.0e-12
            push!(inds, α)
        end
    end
    return inds
end

function q_norm(α::AbstractVector{<:Real}, q::Real)
    s = sum(abs.(α) .^ q)
    return s^(1 / q)
end

"""
    reduced_margin(Λ::Vector{<:Vector{Int}})

Compute the reduced margin of a downward-closed multi-index set ``\\Lambda``:

```math
\\mathcal{R}(\\Lambda)
= \\left\\{\\alpha\\notin\\Lambda:
  \\alpha-e_j\\in\\Lambda\\ \\text{for every }j\\text{ with }\\alpha_j>0
  \\right\\}.
```

# Arguments
- `Λ::Vector{<:Vector{Int}}`: Multi-index set ``\\Lambda``

# Returns
- `Vector{Vector{Int}}`: The reduced margin ``\\mathcal{R}(\\Lambda)``
"""
function reduced_margin(Λ::Vector{<:Vector{Int}})
    # Handle empty input
    if isempty(Λ)
        return Vector{Vector{Int}}()
    end

    d = length(Λ[1])

    # Convert to Set of tuples for O(1) membership tests
    present = Set(tuple(α...) for α in Λ)

    # Generate candidates: all multi-indices one step above elements in Λ
    candidates = Set{NTuple{d, Int}}()
    for β in Λ
        for i in 1:d
            α = copy(β)
            α[i] += 1
            push!(candidates, tuple(α...))
        end
    end

    # A multi-index α is in the reduced margin if:
    # 1. α ∉ Λ (not already in the set)
    # 2. For all j where α[j] > 0: (α - e_j) ∈ Λ (all neighbors are in Λ)
    reduced = Vector{Vector{Int}}()
    for α_tuple in candidates
        # Skip if already in Λ
        α_tuple in present && continue

        α = collect(α_tuple)
        is_reduced = true

        # Check all non-zero coordinates
        for j in 1:d
            if α[j] > 0
                # Check if neighbor (α - e_j) is in Λ
                neighbor = ntuple(i -> i == j ? α[i] - 1 : α[i], d)
                if neighbor ∉ present
                    is_reduced = false
                    break
                end
            end
        end

        is_reduced && push!(reduced, α)
    end

    return reduced
end

# Helper function to convert a matrix of indices to a vector of index vectors
_multivariate_indices(indices::Matrix{Int64}) = collect.(eachrow(indices))
