# Root finder to get inverse of polynomial map components

# Helper function to find bounds for the inverse map
# Based on `inverse` in `IntegratedPositiveFunction.m` from https://github.com/baptistar/ATM
function _inverse_bound(fun::Function)
    # Initial bounds for the root finding
    lower = -1.0
    upper = 1.0

    fa = fun(lower)
    fb = fun(upper)
    # Expand bounds until the root is bracketed
    while fa * fb > 0.0
        delta = 0.5 * (upper - lower)
        if fa > 0
            lower -= delta
            fa = fun(lower)
        elseif fb < 0
            upper += delta
            fb = fun(upper)
        end
    end

    return lower, upper
end


# `hybridRootFindingSolver.m` from https://github.com/baptistar/ATM
function hybridrootfinder(
        f::Function,
        ∂f::Function,
        lower::Real,
        upper::Real;
        xtol::Real = 1.0e-6,
        ftol::Real = 1.0e-6,
        maxiter::Int = 10_000
    )
    x = lower + (upper - lower) / 2
    isfinite(x) || (x = lower / 2 + upper / 2)
    fx, dfx = f(x), ∂f(x)
    # A midpoint may already satisfy the tolerance (including a zero derivative
    # at an exact root). Avoid another residual/derivative pair in that case.
    abs(fx) <= ftol && return x, fx, dfx
    newton_step = fx / dfx
    isfinite(dfx) && !iszero(dfx) && abs(newton_step) <= xtol && return x, fx, dfx
    previous_step = upper - lower

    for _ in 1:maxiter
        # The transport-map residual is increasing. Incorporate the current
        # point before testing whether a Newton step stays inside its bracket.
        if fx < 0.0
            lower = x
        else
            upper = x
        end

        midpoint = lower + (upper - lower) / 2
        # A residual may stop improving at round-off accuracy. Once the bracket
        # is sufficiently narrow, or has no representable interior point, more
        # iterations cannot improve the root's positional accuracy.
        (upper - lower <= xtol || midpoint == lower || midpoint == upper) &&
            return x, fx, dfx

        # Keep endpoint roots reachable, but bisect if Newton leaves the
        # bracket, stalls, or fails to halve the preceding step's magnitude.
        candidate = x - newton_step
        if !isfinite(dfx) || iszero(dfx) || !(lower <= candidate <= upper) ||
                candidate == x || abs(newton_step) > abs(previous_step) / 2
            candidate = midpoint
        end
        previous_step = candidate - x
        x = candidate
        fx, dfx = f(x), ∂f(x)
        abs(fx) <= ftol && return x, fx, dfx
        newton_step = fx / dfx

        if isfinite(dfx) && !iszero(dfx) && abs(newton_step) <= xtol
            return x, fx, dfx
        end
    end

    @warn "Maximum iterations reached in hybridrootfinder"
    return x, fx, dfx
end
