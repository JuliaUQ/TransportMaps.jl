# Gauss-Legendre quadrature to evaluate integral of rectifier

"""
    gaussquadrature(fun::Function, n::Int, a::Real, b::Real)
    gaussquadrature(fun, rule::Tuple{AbstractVector, AbstractVector}, a::Real, b::Real)

Numerically integrates the function `fun` over the interval `[a, b]` using the Gauss-Legendre quadrature rule with `n` points.

# Arguments
- `fun::Function`: The function to integrate. Should accept a single `Float64` argument.
- `n::Int`: The number of quadrature points to use.
- `rule`: Alternatively, reuse the `(points, weights)` returned by `gausslegendre(n)`
  on `[-1, 1]`. The rule is read-only and can be shared by concurrent calls.
- `a::Real`: The lower bound of the integration interval.
- `b::Real`: The upper bound of the integration interval.

# Returns
- `Float64`: The approximate value of the integral of `fun` over `[a, b]`.

"""
function gaussquadrature(
        fun::F,
        n::Int,
        a::Real,
        b::Real
    ) where {F}
    return gaussquadrature(fun, gausslegendre(n), a, b)
end

function gaussquadrature(
        fun::F,
        rule::Tuple{AbstractVector, AbstractVector},
        a::Real,
        b::Real,
    ) where {F}
    points, weights = rule
    length(points) == length(weights) || throw(DimensionMismatch("Quadrature nodes and weights must have equal lengths"))
    scale = 0.5 * (b - a)
    shift = 0.5 * (a + b)
    # Preserve the empty-rule behavior without evaluating the integrand.
    isempty(points) && return dot(weights, fun.(muladd.(points, scale, shift))) * scale
    # Reduce weighted values directly: no transformed-node or function-value arrays.
    return sum(q -> weights[q] * fun(muladd(points[q], scale, shift)), eachindex(points, weights)) * scale
end
