# # Composing Maps
#
# [`ComposedMap`](@ref) combines transport maps into a chain. Layers are stored
# in evaluation order: `ComposedMap(A, B, C)` evaluates `C(B(A(x)))`, and inversion
# applies the inverse layers in reverse order. This lets you combine linear
# standardization with nonlinear maps or append corrections to a fitted map.
#
# ## Constructing a Chain
#
# The following chain applies a linear standardization followed by two polynomial
# maps. These constructors initialize the layers; fitting is a separate operation.

using TransportMaps, Distributions

L = LinearMap([1.0, -1.0], [2.0, 3.0])
P = PolynomialMap(2, 3)
R = PolynomialMap(2, 2, Normal(), Softplus(), HermiteBasis())
C = ComposedMap(L, P, R; forwarddirection = :reference)
#md nothing # hide

x = [0.2, -0.4]
z = evaluate(C, x)           # R(P(L(x)))
x_recovered = inverse(C, z) # L⁻¹(P⁻¹(R⁻¹(z)))
#md nothing # hide

# Matrices hold observations in rows and use the same operations.
X = [0.2 -0.4; 0.5 0.1]
Z = evaluate(C, X)
X_recovered = inverse(C, Z)
#md nothing # hide

# Tuples, vectors, and nested compositions are also accepted:
# `ComposedMap((L, P, R))`, `ComposedMap([L, P, R])`, and
# `ComposedMap(ComposedMap(L, P), R)` have the same evaluation order.
# All layers must have equal dimensions, except the dimension-free identity
# `LinearMap()`. Their domains must also be compatible.
#
# ## Direction and Reference Density
#
# The direction describes the whole chain:
#
# - `forwarddirection = :reference` maps target coordinates to reference coordinates.
# - `forwarddirection = :target` maps reference coordinates to target coordinates.
#
# If the keyword is omitted, directions are inferred from the layers and must
# agree. Linear-only chains default to `:reference`. An explicit direction
# overrides layer metadata without changing evaluation order.
#
# The reference density belongs to the reference endpoint of the whole chain.
# By default, it comes from the last layer with reference metadata for
# `:reference`, or the first such layer for `:target`. Linear-only chains use a
# product standard normal. Pass `reference` explicitly to choose a different
# reference endpoint:

C_with_reference = ComposedMap(L, P, R; forwarddirection = :reference, reference = Normal())
#md nothing # hide

# Univariate distributions become product references; multivariate distributions
# are accepted directly. Inferred direction and reference metadata follow later
# changes to the underlying layers.
#
# ## Sequential Fitting
#
# For a target-to-reference chain `S_previous`, append a newly fitted correction
# `R` so the updated map evaluates `R(S_previous(x))`:
#
# ```julia
# S_next = ComposedMap(S_previous, R; forwarddirection = :reference)
# ```
#
# For a reference-to-target chain `T_previous`, prepend a fitted correction `Q`
# so the updated map evaluates `T_previous(Q(z))`:
#
# ```julia
# T_next = ComposedMap(Q, T_previous; forwarddirection = :target)
# ```
#
# The wrapper composes supplied layers; it does not run an annealing algorithm
# or fit new layers. `C.layers` exposes the tuple of layers. Parameters are shared
# with the supplied maps, so use `deepcopy` when storing a fitted snapshot that
# must survive later changes to those maps.
#
# ## Jacobians and Densities
#
# The forward determinant follows the chain rule, with each layer evaluated at
# its own input. `logabsdetjacobian` sums layer log-determinants to avoid forming
# a potentially overflowing or underflowing product.

determinant = jacobian(C, x)
log_determinant = logabsdetjacobian(C, x)
log_density = logpdf(C, x)
density = pdf(C, x)
#md nothing # hide

# Both directions support `pullback(C, x)` / `pdf(C, x)` for the target-density
# approximation and `logpdf(C, x)` for its log-density. Use
# `pushforward(C, target, z)` to represent a supplied `AbstractMapDensity` target
# in reference coordinates. Jacobians and densities accept individual points
# or matrices with observations in rows.
#
# ## Custom Layers and Existing Code
#
# Custom layers provide `numberdimensions`, `evaluate`, `inverse`, and
# `jacobian(map, x)`. The last operation returns the **forward determinant**.
# Override `logabsdetjacobian` when computing that determinant directly could
# overflow or underflow.
#
# The original `ComposedMap(L, P)` constructor and `.linearmap` / `.polynomialmap`
# accessors remain available for a linear-plus-polynomial pair. Density evaluation
# uses the direction of the whole composition, including for `:target`.
# `jacobian(L)` retains its historical scale-determinant meaning, while
# `jacobian(L, x)` returns the forward determinant used in a composition.
