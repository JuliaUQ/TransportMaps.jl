"""
    Softplus(β::Float64 = 1.0)

Smooth rectifier function

```math
g(\\xi) = \\frac{\\log\\!\\left(1 + \\exp(\\beta \\xi)\\right)}{\\beta}.
```

It is a smooth approximation to ReLU that takes values in ``(0, \\infty)``.
Larger values of ``\\beta`` produce a sharper transition.
"""
struct Softplus <: AbstractRectifierFunction
    β::Float64
    Softplus(β::Float64 = 1.0) = new(β)
end

"""
    second_derivative(rectifier, ξ)

Compute the second derivative of a rectifier. Built-in rectifiers provide analytic
methods; this finite-difference fallback preserves support for custom rectifiers.
"""
function second_derivative(rectifier::AbstractRectifierFunction, ξ::Real)
    return FiniteDiff.finite_difference_derivative(
        value -> derivative(rectifier, value), ξ
    )
end

"""
    (r::Softplus)(ξ)

Evaluate the Softplus rectifier at ``\\xi``.
"""
function (r::Softplus)(ξ)
    return 1 / r.β * log1p(exp(r.β * ξ))
end

"""
    derivative(r::Softplus, ξ)

Compute the derivative of Softplus,

```math
g'(\\xi) = \\frac{1}{1 + \\exp(-\\beta \\xi)}.
```
"""
function derivative(r::Softplus, ξ)
    return 1.0 / (1.0 + exp(-r.β * ξ))  # sigmoid function
end

"""
    second_derivative(r::Softplus, ξ)

Compute the second derivative of Softplus,

```math
g''(\\xi) = \\beta\\,\\sigma(\\beta\\xi)\\left[1 - \\sigma(\\beta\\xi)\\right],
\\qquad
\\sigma(t) = \\frac{1}{1 + \\exp(-t)}.
```
"""
function second_derivative(r::Softplus, ξ::Real)
    sigmoid = derivative(r, ξ)
    return r.β * sigmoid * (1 - sigmoid)
end

"""
    ShiftedELU()

Rectifier combining exponential and linear behavior:

```math
g(\\xi) =
\\begin{cases}
    \\exp(\\xi), & \\xi \\leq 0, \\\\
    \\xi + 1,   & \\xi > 0.
\\end{cases}
```

The result is strictly positive and continuously differentiable.
"""
struct ShiftedELU <: AbstractRectifierFunction
end

"""
    (r::ShiftedELU)(ξ)

Evaluate the ShiftedELU rectifier at ``\\xi``.
"""
function (r::ShiftedELU)(ξ)
    return ξ <= 0 ? exp(ξ) : ξ + 1
end

"""
    derivative(r::ShiftedELU, ξ)

Compute the derivative of ShiftedELU,

```math
g'(\\xi) =
\\begin{cases}
    \\exp(\\xi), & \\xi \\leq 0, \\\\
    1,         & \\xi > 0.
\\end{cases}
```
"""
function derivative(r::ShiftedELU, ξ)
    return ξ <= 0 ? exp(ξ) : 1.0
end

"""
    second_derivative(r::ShiftedELU, ξ)

Compute the piecewise second derivative of ShiftedELU,

```math
g''(\\xi) =
\\begin{cases}
    \\exp(\\xi), & \\xi < 0, \\\\
    0,         & \\xi > 0.
\\end{cases}
```

The second derivative does not exist at ``\\xi = 0``; the symmetric value
``1/2`` is returned there.
"""
function second_derivative(::ShiftedELU, ξ::Real)
    return ξ < 0 ? exp(ξ) : ξ > 0 ? 0.0 : 0.5
end

"""
    IdentityRectifier()

Identity rectifier, ``g(\\xi) = \\xi``. It does not enforce strict positivity, so
use it only when the relevant partial derivatives are positive by construction.
"""
struct IdentityRectifier <: AbstractRectifierFunction
end

"""
    (r::IdentityRectifier)(ξ)

Evaluate the identity rectifier, ``g(\\xi) = \\xi``.
"""
function (r::IdentityRectifier)(ξ)
    return ξ
end

"""
    derivative(r::IdentityRectifier, ξ)

Compute the derivative of `IdentityRectifier`, ``g'(\\xi) = 1``.
"""
function derivative(r::IdentityRectifier, ξ)
    return 1.0
end

"""
    second_derivative(r::IdentityRectifier, ξ)

Compute the second derivative of the identity rectifier, ``g''(\\xi) = 0``.
"""
function second_derivative(::IdentityRectifier, ξ::Real)
    return 0.0
end

"""
    ExpRectifier()

Exponential rectifier, ``g(\\xi) = \\exp(\\xi)``. It ensures strict positivity and
monotonicity, but can produce extreme values for large ``|\\xi|``.
"""
struct ExpRectifier <: AbstractRectifierFunction
end

"""
    (r::ExpRectifier)(ξ)

Evaluate the exponential rectifier, ``g(\\xi) = \\exp(\\xi)``.
"""
function (r::ExpRectifier)(ξ)
    return exp.(ξ)
end

"""
    derivative(r::ExpRectifier, ξ)

Compute the derivative of `ExpRectifier`, ``g'(\\xi) = \\exp(\\xi)``.
"""
function derivative(r::ExpRectifier, ξ)
    return exp.(ξ)
end

"""
    second_derivative(r::ExpRectifier, ξ)

Compute the second derivative of the exponential rectifier,
``g''(\\xi) = \\exp(\\xi)``.
"""
function second_derivative(::ExpRectifier, ξ::Real)
    return exp.(ξ)
end

# Display methods for Softplus
function Base.show(io::IO, s::Softplus)
    print(io, "Softplus(β=$(s.β))")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", s::Softplus)
    println(io, "Softplus:")
    println(io, "  Function: log(1 + exp(βξ)) / β")
    println(io, "  Parameter β: $(s.β)")
    println(io, "  Domain: ℝ")
    println(io, "  Range: (0, ∞)")
    println(io, "  Properties: Smooth approximation to ReLU, always positive")
    println(io, "  Derivative: σ(βξ) = 1/(1 + exp(-βξ)) (sigmoid)")
    return nothing
end

# Display methods for ShiftedELU
function Base.show(io::IO, ::ShiftedELU)
    print(io, "ShiftedELU()")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", ::ShiftedELU)
    println(io, "ShiftedELU:")
    println(io, "  Function: ξ ≤ 0 ? exp(ξ) : ξ + 1")
    println(io, "  Domain: ℝ")
    println(io, "  Range: (0, ∞)")
    println(io, "  Properties: Exponential for negative inputs, linear + 1 for positive")
    println(io, "  Continuous and differentiable everywhere")
    return nothing
end

# Display methods for IdentityRectifier
function Base.show(io::IO, ::IdentityRectifier)
    print(io, "IdentityRectifier()")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", ::IdentityRectifier)
    println(io, "IdentityRectifier:")
    println(io, "  Function: ξ")
    println(io, "  Domain: ℝ")
    println(io, "  Range: ℝ")
    println(io, "  Properties: No transformation, passes input unchanged")
    println(io, "  Warning: May result in non-monotonic transport maps")
    return nothing
end
