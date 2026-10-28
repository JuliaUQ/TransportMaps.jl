using TransportMaps, Distributions, Optim
using CairoMakie, LaTeXStrings, Random, Statistics
set_theme!(transportmap_theme())

b, h, L = 0.2, 0.1, 10.0 # m
E = 210.0e9 # Pa
I = b * h^3 / 12 # m^4
beam_model(θ) = θ[2] * (θ[1] * L)^2 * (3 * L - θ[1] * L) / (6 * E * I)

sequential_prior = PriorMap([Beta(10, 3), Normal(1000, 300)])
sequential_reference = sequential_prior.reference
logprior(θ) = logpdf(sequential_prior, θ)

σ = 0.01 # m
observations = [
    0.04700676518380301,
    0.05567472107255563,
    0.06033689503633009,
    0.035675890874077895,
]

loglikelihood(θ, y) = logpdf(Normal(beam_model(θ), σ), y)
logposterior(θ, k) = logprior(θ) +
    sum(loglikelihood(θ, observations[i]) for i in 1:k; init = 0.0)

function assimilate_observation(
        previous, observation, quadrature; degree = 2,
    )
    # Express the new likelihood in reference coordinates.
    logupdate(z) = logpdf(sequential_reference, z) +
        loglikelihood(evaluate(previous, z), observation)
    update_target = MapTargetDensity(logupdate)

    # Fit a fresh correction, keeping the previous transport fixed.
    correction = PolynomialMap(
        2, degree, Normal(), ShiftedELU(), HermiteBasis(),
    )
    result = optimize!(
        correction, update_target, quadrature;
        options = Optim.Options(iterations = 250, g_tol = 1.0e-7),
        δ = 0.0,
    )

    # The new correction acts first in evaluation order.
    posterior = ComposedMap(
        correction, previous; forwarddirection = :target,
    )
    return posterior, result
end

prior_map = ComposedMap(sequential_prior; forwarddirection = :target)
posterior_maps = ComposedMap[prior_map]
fit_results = []
quadrature = GaussHermiteWeights(4, 2)

for observation in observations
    posterior, result = assimilate_observation(
        posterior_maps[end], observation, quadrature,
    )
    push!(posterior_maps, posterior)
    push!(fit_results, result)
end

reference_samples = randn(MersenneTwister(42), 1200, 2)
reference_map = ComposedMap(LinearMap(2); forwarddirection = :target)
plot_maps = [reference_map; posterior_maps]
stage_samples = map(M -> evaluate(M, reference_samples), plot_maps)

stage_titles = ["Reference", "Prior", ["Posterior $k" for k in eachindex(observations)]...]
transition_labels = [L"P"; [latexstring("T_{", k, "}") for k in eachindex(observations)]]
plot_columns = 2
stage_locations = map(0:(length(plot_maps) - 1)) do k
    row, column = divrem(k, plot_columns)
    (2row + 1, iseven(row) ? 2column + 1 : 2(plot_columns - column) - 1)
end
stage_colors = clamp.(reference_samples[:, 1], -2.5, 2.5)
plot_limits = (0.2, 1.0, 0.0, 2500.0)
stage_limits = [k == 1 ? (-3.0, 3.0, -3.0, 3.0) : plot_limits for k in eachindex(plot_maps)]
stage_density_grids = map(eachindex(plot_maps)) do k
    limits = stage_limits[k]
    xs = range(limits[1], limits[2]; length = 101)
    ys = range(limits[3], limits[4]; length = 101)
    logdensity = [k == 1 ? logpdf(sequential_reference, [x, y]) : logposterior([x, y], k - 2) for x in xs, y in ys]
    density = exp.(logdensity .- maximum(logdensity))
    (; xs, ys, density)
end
function sequential_plot_frame()
    plot_rows = cld(length(plot_maps), plot_columns)
    fig = Figure(size = (640, 300 * plot_rows), figurepadding = (16, 16, 24, 24))
    panels = map(eachindex(plot_maps)) do k
        row, col = stage_locations[k]
        ax = Axis(fig[row, col]; title = stage_titles[k], titlesize = 18, xlabel = k == 1 ? L"z_1" : L"a", ylabel = k == 1 ? L"z_2" : L"F\;[\mathrm{N}]", aspect = 1, xticks = k == 1 ? (-3:3:3) : (0.2:0.4:1.0), yticks = k == 1 ? (-3:3:3) : (0:1000:2000), xgridvisible = true, ygridvisible = true, xminorgridvisible = false, yminorgridvisible = false)
        limits!(ax, stage_limits[k]...)
        ax
    end
    for k in 1:(length(stage_locations) - 1)
        row, column = stage_locations[k]
        next_row, next_column = stage_locations[k + 1]
        arrow = next_row != row ? L"\downarrow" : next_column > column ? L"\longrightarrow" : L"\longleftarrow"
        position = next_row == row ? (row, (column + next_column) ÷ 2) : (row + 1, column)
        transition = GridLayout(fig[position...]; width = 32, height = 40, halign = :center, valign = :center, tellwidth = false, tellheight = false)
        Label(transition[1, 1], transition_labels[k]; fontsize = 16, justification = :center, tellwidth = false, tellheight = false)
        Label(transition[2, 1], arrow; fontsize = 20, tellwidth = false, tellheight = false)
        rowsize!(transition, 1, 18)
        rowsize!(transition, 2, 22)
        rowgap!(transition, 0)
    end
    for column in 1:plot_columns
        colsize!(fig.layout, 2column - 1, Auto(1))
        column < plot_columns && colsize!(fig.layout, 2column, 32)
    end
    for row in 1:plot_rows
        rowsize!(fig.layout, 2row - 1, Auto(1))
        row < plot_rows && rowsize!(fig.layout, 2row, 40)
    end
    colgap!(fig.layout, 4)
    rowgap!(fig.layout, 4)
    return fig, panels
end

fig, panels = sequential_plot_frame()
for k in eachindex(plot_maps)
    grid = stage_density_grids[k]
    contour!(panels[k], grid.xs, grid.ys, grid.density; levels = [0.1, 0.35, 0.7], color = (theme(:textcolor)[], 0.65), linewidth = 1.4)
    scatter!(panels[k], stage_samples[k][:, 1], stage_samples[k][:, 2]; color = stage_colors, colormap = :viridis, markersize = 2.5, colorrange = (-2.5, 2.5))
end

validation_samples = randn(MersenneTwister(123), 2000, 2)
posterior_diagnostics = map(eachindex(observations)) do n
    transport = posterior_maps[n + 1]
    parameters = evaluate(transport, validation_samples)
    logratios = [logposterior(θ, n) for θ in eachrow(parameters)] .+
        logabsdetjacobian(transport, validation_samples) .-
        logpdf(sequential_reference, validation_samples)
    var(logratios) / 2
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
