
struct GaussHermiteKnots <: AbstractQuadratureKnots
end

support(knots::GaussHermiteKnots) = RealInterval(-Inf, Inf)

function (knots::GaussHermiteKnots)(level::Int)
    if level == 0
        return ([0.0], [1.0])
    else
        n = min(2^level + 1, 200)
        return gausshermite(n; normalize=true)
    end
end

struct GaussLegendreKnots <: AbstractQuadratureKnots
    domain::RealInterval{<:Real}

    function GaussLegendreKnots(domain::AbstractVector{<:Real}=[-1, 1])
        return new(RealInterval(domain...))
    end

    function GaussLegendreKnots(domain::RealInterval{<:Real})
        return new(domain)
    end
end

support(knots::GaussLegendreKnots) = knots.domain

function (knots::GaussLegendreKnots)(level::Int)
    if level == 0
        return ([0.0], [1.0])
    else
        n = min(2^level + 1, 200)
        quad = gausslegendre(n)
        transform_to_domain!(knots, quad...)
        return quad
    end
end

# Transform quadrature knots from domain [-1, 1] to [a, b]
function transform_to_domain!(knots::GaussLegendreKnots, quadrature_points, quadrature_weights)
    a, b = extrema(support(knots))
    scale = (b - a) / 2
    shift = (a + b) / 2

    quadrature_points .= scale .* quadrature_points .+ shift
    quadrature_weights .= quadrature_weights ./ 2
end

struct ClenshawCurtisKnots <: AbstractQuadratureKnots
    domain::RealInterval{<:Real}

    function ClenshawCurtisKnots(domain::AbstractVector{<:Real}=[-1, 1])
        return new(RealInterval(domain...))
    end

    function ClenshawCurtisKnots(domain::RealInterval{<:Real})
        return new(domain)
    end
end

support(knots::ClenshawCurtisKnots) = knots.domain

function (knots::ClenshawCurtisKnots)(level::Int)
    if level == 0
        return ([0.0], [1.0])
    else
        quad = clenshaw_curtis_rule(2^level)
        transform_to_domain!(knots, quad...)
        return quad
    end
end

function clenshaw_curtis_rule(n::Int64)
    n ≥ 1 || throw(ArgumentError("n must be ≥ 1"))

    # Chebyshev extrema on [-1, 1]
    x = [cospi(k / n) for k in 0:n]  # cos(π*k/N)

    # Build Vandermonde system A * w = b enforcing exactness for x^m
    # ∫_{-1}^1 x^m dx = 0 for m odd, 2/(m+1) for m even
    A = Matrix{Float64}(undef, n+1, n+1)
    for m in 0:n
        A[m+1, :] .= x .^ m
    end
    b = Float64[(1 + (-1.)^m) / (m + 1) for m in 0:n]

    w = A \ b
    return x, w
end

# Transform quadrature knots from domain [-1, 1] to [a, b]
function transform_to_domain!(knots::ClenshawCurtisKnots, quadrature_points, quadrature_weights)
    a, b = extrema(support(knots))
    scale = (b - a) / 2
    shift = (a + b) / 2

    quadrature_points .= scale .* quadrature_points .+ shift
    quadrature_weights .= quadrature_weights ./ 2
end
