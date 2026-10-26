# # Composing Maps
#
# [`ComposedMap`](@ref) combines transport maps into a chain. Layers are stored
# in evaluation order: `ComposedMap(A, B, C)` evaluates `C(B(A(x)))`, and inversion
# applies the inverse layers in reverse order. This lets you combine linear
# standardization with nonlinear maps or append corrections to a fitted map.
#
# ## Composition and Change of Variables
#
# Let ``T_1,\ldots,T_K:\mathbb{R}^d\to\mathbb{R}^d`` be differentiable,
# invertible maps with compatible domains. Starting from ``u_0=z``, define
#
# ```math
# u_k=T_k(u_{k-1}),\qquad
# C=T_K\circ\cdots\circ T_1,\qquad
# C(z)=u_K.
# ```
#
# Thus, the first argument to `ComposedMap` is the first map applied. Inversion
# reverses the sequence:
#
# ```math
# C^{-1}=T_1^{-1}\circ\cdots\circ T_K^{-1}.
# ```
#
# For a reference density ``\rho``, write ``q_k`` for the density after the
# first ``k`` layers, with ``q_0=\rho``. The change-of-variables formula gives
#
# ```math
# q_k(u)=
# \frac{q_{k-1}(T_k^{-1}(u))}
#      {|\det DT_k(T_k^{-1}(u))|},
# \qquad
# C_\sharp\rho=q_K.
# ```
#
# The notation ``C_\sharp\rho`` denotes the mathematical pushforward: if
# ``Z\sim\rho``, then ``C(Z)\sim q_K``. This is the sampling direction
# `forwarddirection = :target`. For an integrable quantity ``h``,
# ``\mathbb{E}_{q_K}[h(X)]=\mathbb{E}_{\rho}[h(C(Z))]``; reference samples and
# quadrature nodes can therefore be propagated through the same chain.
#
# The Jacobian follows the same intermediate states:
#
# ```math
# DC(z)=DT_K(u_{K-1})\cdots DT_1(u_0),\qquad
# \log|\det DC(z)|=
# \sum_{k=1}^K\log|\det DT_k(u_{k-1})|.
# ```
#
# In particular, each determinant is evaluated at that layer's input. If every
# layer is lower triangular with positive diagonal derivatives, the composition
# retains both properties, since
#
# ```math
# \frac{\partial C^j}{\partial z_j}(z)=
# \prod_{k=1}^K
# \frac{\partial T_k^j}{\partial u_j}(u_{k-1})>0.
# ```
#
# These formulas describe the container's operations independently of how the
# individual layers were constructed or fitted.
#
# ## Constructing a Chain
#
# The following chain applies a linear standardization followed by two polynomial
# maps. These constructors initialize the layers; fitting is a separate operation.

using TransportMaps, Distributions
using CairoMakie, LaTeXStrings, Random # hide
set_theme!(transportmap_theme()) # hide
#md using Main.DocPlots: save_docplot # hide

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
# ## Following Three Maps
#
# Consider three explicit maps, with ``a=1.2``, ``b=0.7``, ``c=0.45``,
# ``\gamma=0.08``, and ``\eta=0.3``:
#
# ```math
# \begin{aligned}
# T_1(z)&=(az_1,\;bz_2),\\
# T_2(u)&=(u_1,\;u_2+c(u_1^2-a^2)),\\
# T_3(v)&=(v_1+\gamma v_1^3,\;v_2+\eta v_1).
# \end{aligned}
# ```
#
# The first layer changes the scales of a standard-normal reference. The second
# is a **banana map**: it bends horizontal bands into parabolas. Subtracting
# ``a^2=\mathbb{E}[u_1^2]`` keeps its second coordinate centered. The third layer
# stretches the first coordinate's tails and adds a linear shear to the second.
# All three are globally invertible and lower triangular.
#
# `LinearMap` evaluates ``(z-\mu)/\sigma``, so reciprocal scales implement
# ``T_1``. For the nonlinear layers, use Hermite expansions with `ShiftedELU()`,
# which satisfies ``g(0)=1`` and ``g(\xi)=1+\xi`` for ``\xi\geq0``.

a, b, c = 1.2, 0.7, 0.45
γ, η = 0.08, 0.3
scale_map = LinearMap(zeros(2), [1 / a, 1 / b])

banana_components = [
    PolynomialMapComponent([[0]], ShiftedELU(), HermiteBasis(), Normal()),
    PolynomialMapComponent([[0, 0], [2, 0]], ShiftedELU(), HermiteBasis(), Normal()),
]
banana_map = PolynomialMap(banana_components)
setcoefficients!(banana_map.components[2], [c * (1 - a^2), c])

stretch_components = [
    PolynomialMapComponent([[1], [3]], ShiftedELU(), HermiteBasis(), Normal()),
    PolynomialMapComponent([[1, 0]], ShiftedELU(), HermiteBasis(), Normal()),
]
stretch_map = PolynomialMap(stretch_components)
setcoefficients!(stretch_map.components[1], [3γ, γ])
setcoefficients!(stretch_map.components[2], [η])

chain = ComposedMap(scale_map, banana_map, stretch_map; forwarddirection = :target)
#md nothing # hide

# Here ``H_2(t)=t^2-1`` and ``H_3(t)=t^3-3t``. The banana component's
# underlying polynomial is ``c(u_1^2-a^2)`` and has zero derivative in ``u_2``,
# so integration adds exactly ``u_2``. The stretch component's polynomial is
# ``\gamma v_1^3``, whose derivative ``3\gamma v_1^2`` is nonnegative; its
# rectified integral is exactly ``v_1+\gamma v_1^3``. No optimization is required
# for these illustrative layers.
#
# ### Samples and Intermediate Densities
#
# Draw reference samples once and evaluate the prefixes of the chain. Each panel
# below shows the **same observations**, with color recording their original
# first reference coordinate. Follow the arrows clockwise from the upper left.
# Contours show the density of each prefix, including its Jacobian factor.

reference_samples = randn(MersenneTwister(42), 1200, 2)
prefixes = (
    ComposedMap(LinearMap(2); forwarddirection = :target),
    ComposedMap(scale_map; forwarddirection = :target),
    ComposedMap(scale_map, banana_map; forwarddirection = :target),
    chain,
)
stage_samples = map(M -> evaluate(M, reference_samples), prefixes)
#md nothing # hide

stage_titles = ("Reference", "1. Scale", "2. Banana", "3. Stretch") # hide
stage_locations = ((1, 1), (1, 3), (3, 3), (3, 1)) # hide
stage_colors = clamp.(reference_samples[:, 1], -2.5, 2.5) # hide
grid_coordinate = range(-2.5, 2.5; length = 81) # hide
grid_boundary = vcat( # hide
    hcat(grid_coordinate, fill(-2.5, length(grid_coordinate))), # hide
    hcat(grid_coordinate, fill(2.5, length(grid_coordinate))), # hide
    hcat(fill(-2.5, length(grid_coordinate)), grid_coordinate), # hide
    hcat(fill(2.5, length(grid_coordinate)), grid_coordinate), # hide
) # hide
stage_limits = map(prefixes) do M # hide
    boundary = evaluate(M, grid_boundary) # hide
    lo, hi = vec(minimum(boundary; dims = 1)), vec(maximum(boundary; dims = 1)) # hide
    margin = 0.08 .* (hi .- lo) # hide
    (lo[1] - margin[1], hi[1] + margin[1], lo[2] - margin[2], hi[2] + margin[2]) # hide
end # hide
stage_density_grids = map(zip(prefixes, stage_limits)) do (M, limits) # hide
    xs = range(limits[1], limits[2]; length = 65) # hide
    ys = range(limits[3], limits[4]; length = 65) # hide
    points = [Float64[x, y] for x in xs, y in ys] # hide
    density = reshape(pdf(M, reduce(vcat, permutedims.(vec(points)))), length(xs), length(ys)) # hide
    (; xs, ys, density) # hide
end # hide
function chain_plot_frame() # hide
    fig = Figure(size = (600, 650)) # hide
    panels = map(eachindex(prefixes)) do k # hide
        row, col = stage_locations[k] # hide
        ax = Axis(fig[row, col]; title = stage_titles[k], xlabel = L"u_{%$(k - 1),1}", ylabel = L"u_{%$(k - 1),2}", aspect = DataAspect()) # hide
        limits!(ax, stage_limits[k]...) # hide
        ax # hide
    end # hide
    Label(fig[1, 2], L"\longrightarrow"; fontsize = 24, tellwidth = false, tellheight = false) # hide
    Label(fig[2, 3], L"\downarrow"; fontsize = 24, tellwidth = false, tellheight = false) # hide
    Label(fig[3, 2], L"\longleftarrow"; fontsize = 24, tellwidth = false, tellheight = false) # hide
    for i in (1, 3) # hide
        colsize!(fig.layout, i, Relative(0.5)) # hide
        rowsize!(fig.layout, i, Relative(0.5)) # hide
    end # hide
    colsize!(fig.layout, 2, 22) # hide
    rowsize!(fig.layout, 2, 22) # hide
    colgap!(fig.layout, 8) # hide
    rowgap!(fig.layout, 10) # hide
    return fig, panels # hide
end # hide

#md save_docplot("composed-map-samples.svg") do # hide
fig, panels = chain_plot_frame() # hide
for k in eachindex(prefixes) # hide
    grid = stage_density_grids[k] # hide
    contour!(panels[k], grid.xs, grid.ys, grid.density; levels = [0.1, 0.35, 0.7] .* maximum(grid.density), color = (theme(:textcolor)[], 0.65), linewidth = 1.4) # hide
    scatter!(panels[k], stage_samples[k][:, 1], stage_samples[k][:, 2]; color = stage_colors, colormap = :viridis, markersize = 2.5, colorrange = (-2.5, 2.5)) # hide
end # hide
#md     fig # hide
#md end # hide
# ![The same reference samples passing through scaling, a banana map, and a nonlinear stretch, with intermediate density contours](composed-map-samples.svg)
#
# The axes focus on the image of the reference square ``[-2.5,2.5]^2`` with
# a small margin; samples outside those view limits are clipped. Each panel
# uses its own coordinate limits. Colors identify corresponding observations
# across panels rather than density values.
#
# ### Deformation of a Reference Grid
#
# The second figure follows the same regular reference grid through all three
# prefixes. Each panel is the image of the **original grid under the cumulative
# map**, rather than a newly generated grid in that panel's coordinates.

#md save_docplot("composed-map-grid.svg") do # hide
fig, panels = chain_plot_frame() # hide
for k in eachindex(prefixes) # hide
    mappingplot!(panels[k], prefixes[k], (grid_coordinate, grid_coordinate); gridlines = 9, color = :steelblue, linewidth = 1.5) # hide
end # hide
#md     fig # hide
#md end # hide
# ![A reference grid propagated through the three-map chain, showing the banana bend and subsequent nonlinear stretch](composed-map-grid.svg)
#
# For this example the final map and determinant are available explicitly:
#
# ```math
# \begin{aligned}
# C(z)&=\left(
# az_1+\gamma a^3z_1^3,\;
# bz_2+ca^2(z_1^2-1)+\eta az_1
# \right),\\
# \det DC(z)&=ab(1+3\gamma a^2z_1^2)>0.
# \end{aligned}
# ```
#
# The banana layer has determinant one: it bends the contours while preserving
# area. Scaling changes area by ``ab``, and the last layer adds a spatially
# varying factor ``1+3\gamma v_1^2``. Together these explain the intermediate
# density contours as well as the grid deformation.
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
# ### Sequential Bayesian Updates
#
# Rubio, Chamoin, and Louf [rubio2021](@cite), equations (18)–(19), use
# compositions of low-order maps for sequential inference. With likelihood
# ``\ell_n`` and previous posterior ``\pi_{n-1}``, the next target satisfies
# ``\pi_n(x)\propto\ell_n(x)\pi_{n-1}(x)``. If ``T_{n-1}`` transports
# ``\rho`` exactly to ``\pi_{n-1}``, its new reference-space target is
#
# ```math
# r_n(z)\propto\ell_n(T_{n-1}(z))\,\rho(z).
# ```
#
# Fit ``Q_n`` from ``\rho`` to ``r_n`` and set
# ``T_n=T_{n-1}\circ Q_n``: the new correction acts first. In the constructor's
# evaluation order, this is `ComposedMap(Q_n, T_previous; forwarddirection = :target)`.
# Their written order ``M_1\circ\cdots\circ M_n`` therefore corresponds to
# constructor arguments `(M_n, ..., M_1)`.
#
# For an approximate previous transport, the exact pulled-back target is instead
#
# ```math
# r_n(z)\propto
# \ell_n(T_{n-1}(z))\,
# \pi_{n-1}(T_{n-1}(z))\,
# |\det DT_{n-1}(z)|.
# ```
#
# Replacing its last two factors by ``\rho(z)`` neglects the previous transport's
# approximation error. A reference-to-target correction can be fitted by
# minimizing the reverse Kullback–Leibler objective
#
# ```math
# \mathcal{J}(Q)=
# \mathbb{E}_{Z\sim\rho}\!\left[
# \log\rho(Z)-\log\bar r_n(Q(Z))-\log|\det DQ(Z)|
# \right],
# ```
#
# where ``\bar r_n`` is unnormalized; its unknown normalizing constant does
# not affect the minimizer. The container represents the resulting composition,
# while choosing likelihoods, fitting corrections, and checking approximation
# accuracy remain part of the sequential inference algorithm.
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
#
# In the sampling direction, the target-density approximation satisfies
#
# ```math
# \log q(C(z))=\log\rho(z)-\log|\det DC(z)|.
# ```
#
# For a target-to-reference map ``S`` (`forwarddirection = :reference`), the
# corresponding expression at a target point is
#
# ```math
# \log q(x)=\log\rho(S(x))+\log|\det DS(x)|.
# ```

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
