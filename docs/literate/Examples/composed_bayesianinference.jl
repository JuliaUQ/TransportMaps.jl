# # Beam: Sequential Inference with Composed Maps
#
# This example presents sequential Bayesian inference for load position and force on a beam
# from repeated tip-displacement readings. A fixed prior transformation and fitted corrections
# assimilate the measurements one at a time. See [Composing Maps](@ref)
# for the composition convention, map directions, and density evaluation.

using TransportMaps, Distributions, Optim
using CairoMakie, LaTeXStrings, Random, Statistics # hide
set_theme!(transportmap_theme()) # hide
#md using Main.DocPlots: save_docplot # hide

# ## Beam with an Unknown Load
#
# We adapt the beam example from [grashorn2024](@cite) to sequential inference.
# A cantilever beam is clamped at one end. A point
# force ``F`` acts at distance ``aL`` from the clamp, where ``a`` is the
# dimensionless relative load position and ``L`` is the beam length. We want
# to infer the two fixed but unknown physical parameters ``\theta=(a,F)``.
#
# ![Cantilever beam with a point force F at distance aL from the clamp and a measured tip displacement](../assets/beam.svg)
#
# The beam geometry and material properties are known: width ``b=0.2\,\mathrm{m}``,
# height ``h=0.1\,\mathrm{m}``, length ``L=10\,\mathrm{m}``, and Young's
# modulus ``E=210\,\mathrm{GPa}``. The second moment of area is ``I=bh_b^3/12``.
# The deterministic forward model ``\mathcal{M}(\theta)`` predicts the **noiseless tip
# displacement**, in metres, for a given load position and force:
#
# ```math
# \mathcal{M}(\theta)=\mathcal{M}(a,F)=\frac{F(aL)^2(3L-aL)}{6EI}.
# ```

b, h, L = 0.2, 0.1, 10.0 # m
E = 210.0e9 # Pa
I = b * h^3 / 12 # m^4
beam_model(θ) = θ[2] * (θ[1] * L)^2 * (3 * L - θ[1] * L) / (6 * E * I)
#md nothing # hide

# For the prior distributions ``a\sim\operatorname{Beta}(10,3)`` and
# ``F\sim\mathcal{N}(1000\,\mathrm{N},(300\,\mathrm{N})^2)`` are chosen. A fixed
# [`PriorMap`](@ref) layer ``P`` maps the standard-normal reference
# ``z\sim\rho=\mathcal{N}(0,I_2)`` into these physical prior coordinates.

sequential_prior = PriorMap([Beta(10, 3), Normal(1000, 300)])
sequential_reference = sequential_prior.reference
logprior(θ) = logpdf(sequential_prior, θ)
#md nothing # hide

# ## Noisy Measurements and Sequential Observations
#
# ``Y_i`` denotes the random tip-displacement reading in the ``i``th repeated
# measurement of the **same** loaded beam. The measurement error is modelled as
#
# ```math
# Y_i=\mathcal{M}(\theta)+\varepsilon_i,\qquad
# \varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,\sigma^2),
# ```
#
# with a noise standard deviation ``\sigma=0.01\,\mathrm{m}``.
# Updating is performed given the data:

σ = 0.01 # m
observations = [
    0.04700676518380301,
    0.05567472107255563,
    0.06033689503633009,
    0.035675890874077895,
]
#md nothing #hide

# For the updating, a **sequential** approach is used, fitting one new map after each observation.
# At update ``k``, only ``y_k`` is new.

loglikelihood(θ, y) = logpdf(Normal(beam_model(θ), σ), y)
logposterior(θ, k) = logprior(θ) +
    sum(loglikelihood(θ, observations[i]) for i in 1:k; init = 0.0)
#md nothing # hide

# The likelihood of the ``k``th observation and the resulting posterior are
#
# ```math
# \ell_k(\theta)=\mathcal{N}(y_k;\mathcal{M}(\theta),\sigma^2),\qquad
# \pi_k(\theta)=p(\theta\mid y_1,\ldots,y_k)
# \propto\pi_{k-1}(\theta)\,\ell_k(\theta).
# ```
#
# ## Fit One New Correction per Observation
#
# Starting with ``T_0=P``, fit a correction ``Q_k`` for each new reading
# ``y_k`` using the sequential construction of [rubio2021](@cite):
#
# ```math
# r_k(z)\propto\rho(z)\,\ell_k(T_{k-1}(z)),\qquad
# T_k=T_{k-1}\circ Q_k.
# ```

function assimilate_observation(
        previous, observation, quadrature; degree = 2,
    )
    ## Express the new likelihood in reference coordinates.
    logupdate(z) = logpdf(sequential_reference, z) +
        loglikelihood(evaluate(previous, z), observation)
    update_target = MapTargetDensity(logupdate)

    ## Fit a fresh correction, keeping the previous transport fixed.
    correction = PolynomialMap(
        2, degree, Normal(), ShiftedELU(), HermiteBasis(),
    )
    result = optimize!(
        correction, update_target, quadrature;
        options = Optim.Options(iterations = 250, g_tol = 1.0e-7),
        δ = 0.0,
    )

    ## The new correction acts first in evaluation order.
    posterior = ComposedMap(
        correction, previous; forwarddirection = :target,
    )
    return posterior, result
end
#md nothing # hide

# Initialize the prior transport and a shared reference quadrature:

prior_map = ComposedMap(sequential_prior; forwarddirection = :target)
posterior_maps = ComposedMap[prior_map]
fit_results = []
quadrature = GaussHermiteWeights(4, 2)
#md nothing # hide

# Fit one correction per reading and store each posterior map:

for observation in observations
    posterior, result = assimilate_observation(
        posterior_maps[end], observation, quadrature,
    )
    push!(posterior_maps, posterior)
    push!(fit_results, result)
end
#md nothing # hide

# The final transport is ``T_4=P\circ Q_1\circ\cdots\circ Q_4``. Each update
# inherits approximation error from earlier fits; the diagnostic below checks
# the accumulated posterior against all observations.
#
# ### Distribution Snapshots
#
# Reuse the same reference draws to show ``z``, ``P(z)``, and ``T_k(z)``.
# Colors identify corresponding draws. The arrows label the resulting
# transports; the panels are distribution snapshots, not successive layer
# evaluations. Each posterior map starts from the original reference draws.

reference_samples = randn(MersenneTwister(42), 1200, 2)
reference_map = ComposedMap(LinearMap(2); forwarddirection = :target)
plot_maps = [reference_map; posterior_maps]
stage_samples = map(M -> evaluate(M, reference_samples), plot_maps)
#md nothing # hide

stage_titles = ["Reference", "Prior", ["Posterior $k" for k in eachindex(observations)]...] # hide
transition_labels = [L"P"; [latexstring("T_{", k, "}") for k in eachindex(observations)]] # hide
plot_columns = 2 # hide
stage_locations = map(0:(length(plot_maps) - 1)) do k # hide
    row, column = divrem(k, plot_columns) # hide
    (2row + 1, iseven(row) ? 2column + 1 : 2(plot_columns - column) - 1) # hide
end # hide
stage_colors = clamp.(reference_samples[:, 1], -2.5, 2.5) # hide
plot_limits = (0.2, 1.0, 0.0, 2500.0) # hide
stage_limits = [k == 1 ? (-3.0, 3.0, -3.0, 3.0) : plot_limits for k in eachindex(plot_maps)] # hide
stage_density_grids = map(eachindex(plot_maps)) do k # hide
    limits = stage_limits[k] # hide
    xs = range(limits[1], limits[2]; length = 101) # hide
    ys = range(limits[3], limits[4]; length = 101) # hide
    logdensity = [k == 1 ? logpdf(sequential_reference, [x, y]) : logposterior([x, y], k - 2) for x in xs, y in ys] # hide
    density = exp.(logdensity .- maximum(logdensity)) # hide
    (; xs, ys, density) # hide
end # hide
function sequential_plot_frame() # hide
    plot_rows = cld(length(plot_maps), plot_columns) # hide
    fig = Figure(size = (640, 300 * plot_rows), figurepadding = (16, 16, 24, 24)) # hide
    panels = map(eachindex(plot_maps)) do k # hide
        row, col = stage_locations[k] # hide
        ax = Axis(fig[row, col]; title = stage_titles[k], titlesize = 18, xlabel = k == 1 ? L"z_1" : L"a", ylabel = k == 1 ? L"z_2" : L"F\;[\mathrm{N}]", aspect = 1, xticks = k == 1 ? (-3:3:3) : (0.2:0.4:1.0), yticks = k == 1 ? (-3:3:3) : (0:1000:2000), xgridvisible = true, ygridvisible = true, xminorgridvisible = false, yminorgridvisible = false) # hide
        limits!(ax, stage_limits[k]...) # hide
        ax # hide
    end # hide
    for k in 1:(length(stage_locations) - 1) # hide
        row, column = stage_locations[k] # hide
        next_row, next_column = stage_locations[k + 1] # hide
        arrow = next_row != row ? L"\downarrow" : next_column > column ? L"\longrightarrow" : L"\longleftarrow" # hide
        position = next_row == row ? (row, (column + next_column) ÷ 2) : (row + 1, column) # hide
        transition = GridLayout(fig[position...]; width = 32, height = 40, halign = :center, valign = :center, tellwidth = false, tellheight = false) # hide
        Label(transition[1, 1], transition_labels[k]; fontsize = 16, justification = :center, tellwidth = false, tellheight = false) # hide
        Label(transition[2, 1], arrow; fontsize = 20, tellwidth = false, tellheight = false) # hide
        rowsize!(transition, 1, 18) # hide
        rowsize!(transition, 2, 22) # hide
        rowgap!(transition, 0) # hide
    end # hide
    for column in 1:plot_columns # hide
        colsize!(fig.layout, 2column - 1, Auto(1)) # hide
        column < plot_columns && colsize!(fig.layout, 2column, 32) # hide
    end # hide
    for row in 1:plot_rows # hide
        rowsize!(fig.layout, 2row - 1, Auto(1)) # hide
        row < plot_rows && rowsize!(fig.layout, 2row, 40) # hide
    end # hide
    colgap!(fig.layout, 4) # hide
    rowgap!(fig.layout, 4) # hide
    return fig, panels # hide
end # hide

#md save_docplot("composed-map-samples.svg") do # hide
fig, panels = sequential_plot_frame() # hide
for k in eachindex(plot_maps) # hide
    grid = stage_density_grids[k] # hide
    contour!(panels[k], grid.xs, grid.ys, grid.density; levels = [0.1, 0.35, 0.7], color = (theme(:textcolor)[], 0.65), linewidth = 1.4) # hide
    scatter!(panels[k], stage_samples[k][:, 1], stage_samples[k][:, 2]; color = stage_colors, colormap = :viridis, markersize = 2.5, colorrange = (-2.5, 2.5)) # hide
end # hide
#md     fig # hide
#md end # hide
# ![Standard-normal reference, beam parameter prior, and four beam posterior approximations after 1, 2, 3, and 4 readings over exact Bayesian density contours](composed-map-samples.svg)

# ### Checking the Accumulated Approximation
#
# Evaluate the log-density ratio on independent reference draws. It is
# constant for an exact transport, so its variance measures approximation error:
#
# ```math
# V_k=\tfrac12\operatorname{Var}_{Z\sim\rho}\!\left[
# \log\bar\pi_k(T_k(Z))+\log|\det \mathbf{J}_{T_k}(Z)|-\log\rho(Z)
# \right].
# ```

validation_samples = randn(MersenneTwister(123), 2000, 2)
posterior_diagnostics = map(eachindex(observations)) do n
    transport = posterior_maps[n + 1]
    parameters = evaluate(transport, validation_samples)
    logratios = [logposterior(θ, n) for θ in eachrow(parameters)] .+
        logabsdetjacobian(transport, validation_samples) .-
        logpdf(sequential_reference, validation_samples)
    var(logratios) / 2
end
#md nothing # hide

#md using Markdown # hide
#md Markdown.parse("| Total observations assimilated | Variance diagnostic |\n|:--|--:|\n" * join(["| $n | $(round(posterior_diagnostics[n]; sigdigits = 3)) |" for n in eachindex(observations)], "\n")) # hide
#
# Smaller values indicate a better fit to the accumulated posterior, though
# they do not guarantee global accuracy.
#
