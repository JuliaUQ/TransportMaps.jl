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

using TransportMaps, Distributions, Optim
using CairoMakie, LaTeXStrings, Random, Statistics # hide
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
# ## Sequential Inference from Three Observations
#
# Start with a standard-normal prior on two unknown parameters,
# ``\theta=(\theta_1,\theta_2)\sim\pi_0=\mathcal{N}(0,I_2)``. A nonlinear
# sensor observes
#
# ```math
# Y_n=h(\theta)+\varepsilon_n,\qquad
# h(\theta)=\theta_2-\beta\theta_1^2,\qquad
# \varepsilon_n\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,\sigma^2).
# ```
#
# Take ``\beta=0.65``, ``\sigma=0.6``, and assimilate the observed values
# ``y_1=-0.4``, ``y_2=-0.2``, and ``y_3=-0.5`` one at a time. Each observation
# gives a likelihood ``\ell_n(\theta)=\mathcal{N}(y_n;h(\theta),\sigma^2)``.
# The posterior after the first ``n`` observations is
#
# ```math
# \pi_n(\theta)\propto\pi_0(\theta)\prod_{j=1}^{n}\ell_j(\theta),
# \qquad
# \log\bar\pi_n(\theta)=
# -\frac{\|\theta\|^2}{2}
# -\sum_{j=1}^{n}\frac{(y_j-h(\theta))^2}{2\sigma^2}+\mathrm{const}.
# ```
#
# The likelihood concentrates around the parabola
# ``\theta_2=\beta\theta_1^2+y_n``, so the first posterior has a **banana shape**.
# Later observations refine that same posterior. This small model illustrates
# the paper's sequential construction [rubio2021](@cite); it uses its own
# nonlinear sensor rather than the paper's mechanical simulator.

β, σ = 0.65, 0.6
observations = [-0.4, -0.2, -0.5]
logprior(θ) = -sum(abs2, θ) / 2 - log(2π)
sensor(θ) = θ[2] - β * θ[1]^2
loglikelihood(θ, y) = logpdf(Normal(0, σ), y - sensor(θ))
logposterior(θ, n) = logprior(θ) +
    sum(loglikelihood(θ, observations[j]) for j in 1:n; init = 0.0)
#md nothing # hide

# ### Fit One New Correction per Observation
#
# Let ``T_{n-1}`` map the fixed standard-normal reference ``\rho=\pi_0`` to
# the previous posterior approximation. Following equation (19) of the paper,
# fit a new reference-space correction ``Q_n`` to
#
# ```math
# r_n(z)\propto\rho(z)\,\ell_n(T_{n-1}(z)),\qquad
# T_n=T_{n-1}\circ Q_n.
# ```
#
# The old transport is kept fixed. Only the current likelihood enters this
# correction target: earlier observations are already represented by
# ``T_{n-1}``. Because the new correction acts **before** the old transport,
# it is prepended in the constructor's evaluation order.

function assimilate_observation(previous, observation, quadrature; degree = 5)
    update_target = MapTargetDensity(
        z -> logprior(z) + loglikelihood(evaluate(previous, z), observation),
    )
    correction = PolynomialMap(2, degree, Normal(), ShiftedELU(), HermiteBasis())
    result = optimize!(
        correction, update_target, quadrature;
        options = Optim.Options(iterations = 250, g_tol = 1.0e-7), δ = 0.0,
    )
    posterior = ComposedMap(correction, previous; forwarddirection = :target)
    return posterior, result
end

prior_map = ComposedMap(LinearMap(2); forwarddirection = :target)
posterior_maps = ComposedMap[prior_map]
fit_results = []
quadrature = GaussHermiteWeights(4, 2)

for observation in observations
    posterior, result = assimilate_observation(posterior_maps[end], observation, quadrature)
    push!(posterior_maps, posterior)
    push!(fit_results, result)
end
#md nothing # hide

# There are three fitted corrections. Starting from the identity prior map,
# ``T_1=Q_1``, ``T_2=Q_1\circ Q_2``, and
# ``T_3=Q_1\circ Q_2\circ Q_3``. The stored maps are snapshots after each data
# update; they are not the intermediate coordinates of a single forward
# evaluation of ``T_3``. Each fit starts with a fresh identity-initialized
# polynomial map: zero coefficients and `ShiftedELU()` give ``g(0)=1``.
# The quadrature integrates the reference density and uses no posterior samples.
#
# The likelihood-only update is exact when the previous transport is exact.
# With fitted maps it also carries forward their approximation error; the
# [Sequential Fitting](@ref) section below gives the full pulled-back target.
# Here we check the approximation against the posterior using **all** observed
# likelihoods, rather than treating the fitted density as ground truth.
#
# ### Prior and Successive Posteriors
#
# Draw reference samples once. In panel ``n``, plot ``T_n(z_i)`` for those same
# reference draws; color identifies the original first reference coordinate.
# The arrows denote assimilation of observations 1, 2, and 3, clockwise from
# the prior in the upper left. Each posterior map is evaluated on the original
# reference samples, not on the preceding panel's transported samples.

reference_samples = randn(MersenneTwister(42), 1200, 2)
stage_samples = map(M -> evaluate(M, reference_samples), posterior_maps)
#md nothing # hide

stage_titles = ("Prior", "Posterior 1", "Posterior 2", "Posterior 3") # hide
stage_locations = ((1, 1), (1, 3), (3, 3), (3, 1)) # hide
stage_colors = clamp.(reference_samples[:, 1], -2.5, 2.5) # hide
plot_limits = (-2.7, 2.7, -2.0, 3.0) # hide
stage_density_grids = map(0:length(observations)) do n # hide
    xs = range(plot_limits[1], plot_limits[2]; length = 101) # hide
    ys = range(plot_limits[3], plot_limits[4]; length = 101) # hide
    logdensity = [logposterior([x, y], n) for x in xs, y in ys] # hide
    density = exp.(logdensity .- maximum(logdensity)) # hide
    (; xs, ys, density) # hide
end # hide
function sequential_plot_frame() # hide
    fig = Figure(size = (600, 650)) # hide
    panels = map(eachindex(posterior_maps)) do k # hide
        row, col = stage_locations[k] # hide
        ax = Axis(fig[row, col]; title = stage_titles[k], xlabel = L"\theta_1", ylabel = L"\theta_2", aspect = DataAspect()) # hide
        limits!(ax, plot_limits...) # hide
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
fig, panels = sequential_plot_frame() # hide
for k in eachindex(posterior_maps) # hide
    grid = stage_density_grids[k] # hide
    contour!(panels[k], grid.xs, grid.ys, grid.density; levels = [0.1, 0.35, 0.7], color = (theme(:textcolor)[], 0.65), linewidth = 1.4) # hide
    scatter!(panels[k], stage_samples[k][:, 1], stage_samples[k][:, 2]; color = stage_colors, colormap = :viridis, markersize = 2.5, colorrange = (-2.5, 2.5)) # hide
end # hide
#md     fig # hide
#md end # hide
# ![Prior samples and successive posterior approximations after observations 1, 2, and 3, over exact Bayesian density contours](composed-map-samples.svg)
#
# Contours come directly from ``\bar\pi_n``, at 10%, 35%, and 70% of its peak
# on the plotting grid. Their levels do not require the normalizing constant.
# All four panels use the same physical coordinates and limits, so contraction
# across observations is visible. Samples outside the displayed limits are
# clipped; colors indicate corresponding reference draws, not density values.
#
# ### How Each Data Update Changes the Transport
#
# Transport the same regular reference grid with each stored map ``T_n``. The
# first update bends it toward the banana posterior; the next two updates
# refine the posterior as new observations arrive. Each panel shows a complete
# reference-to-parameter map after the indicated observation, rather than the
# isolated correction ``Q_n``.

grid_coordinate = range(-2.5, 2.5; length = 81) # hide
#md save_docplot("composed-map-grid.svg") do # hide
fig, panels = sequential_plot_frame() # hide
for k in eachindex(posterior_maps) # hide
    mappingplot!(panels[k], posterior_maps[k], (grid_coordinate, grid_coordinate); gridlines = 9, color = :steelblue, linewidth = 1.5) # hide
end # hide
#md     fig # hide
#md end # hide
# ![A reference grid mapped to the prior and successive posteriors as three observations are assimilated](composed-map-grid.svg)
#
# ### Checking the Accumulated Approximation
#
# Use independent reference draws to evaluate the full-posterior log-density
# ratio. It is constant for an exact transport, even when the target density
# is known only up to normalization:
#
# ```math
# V_n=\tfrac12\operatorname{Var}_{Z\sim\rho}\!\left[
# \log\bar\pi_n(T_n(Z))+\log|\det DT_n(Z)|-\log\rho(Z)
# \right].
# ```

validation_samples = randn(MersenneTwister(123), 2000, 2)
posterior_diagnostics = map(eachindex(observations)) do n
    transport = posterior_maps[n + 1]
    parameters = evaluate(transport, validation_samples)
    logratios = [logposterior(θ, n) for θ in eachrow(parameters)] .+
        logabsdetjacobian(transport, validation_samples) .-
        [logprior(z) for z in eachrow(validation_samples)]
    var(logratios) / 2
end
#md nothing # hide

#md using Markdown # hide
#md Markdown.parse("| Observations assimilated | Variance diagnostic |\n|:--|--:|\n" * join(["| $n | $(round(posterior_diagnostics[n]; sigdigits = 3)) |" for n in eachindex(observations)], "\n")) # hide
#
# A small value supports the displayed approximation but does not guarantee
# global accuracy. The diagnostic checks all likelihoods, so it also measures
# error carried over from earlier fitted corrections.
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
