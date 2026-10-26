# Maps

## Index

```@index
Pages = ["maps.md"]
```

## Types

```@docs
PolynomialMap
PolynomialMapComponent
LinearMap
LaplaceMap
ComposedMap
```

## Functions

```@docs
evaluate
inverse
pushforward
pullback
logabsdetjacobian
setcoefficients!
getcoefficients
setparameters!
numbercoefficients
numberdimensions
getmultiindexsets
conditional_density
conditional_sample
multivariate_conditional_density
multivariate_conditional_sample
```

## Composing fitted maps

`ComposedMap` stores layers in evaluation order. For example,

```julia
using TransportMaps, Distributions

L = LinearMap([1.0, -1.0], [2.0, 3.0])
P = PolynomialMap(2, 3)
R = PolynomialMap(2, 2, Normal(), Softplus(), HermiteBasis())
C = ComposedMap(L, P, R; forwarddirection=:reference)

x = [0.2, -0.4]
z = evaluate(C, x)           # R(P(L(x)))
x_recovered = inverse(C, z) # L⁻¹(P⁻¹(R⁻¹(z)))
log_density = logpdf(C, x)
log_jacobian = logabsdetjacobian(C, x)
```

These constructors initialize maps; fitting remains a separate operation.
Tuples, vectors, and nested compositions are also accepted:
`ComposedMap((L, P, R))`, `ComposedMap([L, P, R])`, or
`ComposedMap(ComposedMap(L, P), R)`.

For sequential fitting, append a newly fitted target-to-reference correction
with `ComposedMap(S_previous, R; forwarddirection=:reference)`. For the opposite
direction, prepend the correction:
`ComposedMap(Q, T_previous; forwarddirection=:target)` evaluates `T_previous(Q(z))`.
The wrapper does not run an annealing algorithm or fit new layers itself.

`C.layers` exposes the tuple of layers. Parameters are shared with the supplied
maps, so use `deepcopy` when storing snapshots that must survive later fitting.
Custom layers must provide `numberdimensions`, `evaluate`, `inverse`, and
`jacobian(map, x)`; the last operation must return the **forward determinant**.
Override `logabsdetjacobian` when a determinant could overflow or underflow.
Layers must have compatible domains as well as equal dimensions.

`forwarddirection=:reference` means the whole composition maps target to
reference; `:target` means reference to target. If omitted, directions are
inferred from the layers and must agree. Linear-only compositions default to
`:reference`. Set the keyword explicitly when layer metadata differs.
The default reference comes from the last layer with reference metadata for
`:reference`, or the first for `:target`; linear-only maps use a product standard
normal. An explicit `reference` describes the reference endpoint of the whole
composition. Univariate distributions become product references, and
multivariate distributions are accepted directly.

Both directions support `pullback(C, x)` / `pdf(C, x)` for the target density,
`logpdf(C, x)`, and `pushforward(C, target, z)` for a supplied target density
represented in reference coordinates. Jacobians and densities accept points
or matrices with samples in rows.

The original `ComposedMap(L, P)` API and `.linearmap` / `.polynomialmap` accessors
remain available for a linear-plus-polynomial pair. Density evaluation now uses
the direction of the whole composition, including for `:target`; the old
linear scaling shortcut was only appropriate for `:reference` maps.
`jacobian(L)` retains its historical scale-determinant meaning, while
`jacobian(L, x)` returns the forward determinant used in a composition.
