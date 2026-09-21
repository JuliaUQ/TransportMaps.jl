using TransportMaps, Distributions, CairoMakie, LaTeXStrings
set_theme!(transportmap_theme())

distribution = Logistic(0, 1)
x = range(-5, 5; length = 300)
density_fig, density_ax, plt = lines(
    x, pdf.(distribution, x);
    axis = (; xlabel = L"x", ylabel = L"p(x)"),
    figure = (; size = (600, 300)),
)

target = MapTargetDensity(x -> logpdf(distribution, x[1]))
uniform_reference = Uniform(0, 1)
normal_reference = Normal()
uniform_map = PolynomialMap(1, 9, uniform_reference, Softplus(), ShiftedLegendreBasis())
normal_map = PolynomialMap(1, 9, normal_reference, Softplus(), LinearizedHermiteBasis())
uniform_result = optimize!(uniform_map, target, GaussLegendreWeights(8, uniform_map))
normal_result = optimize!(normal_map, target, GaussHermiteWeights(8, normal_map))

X = reshape(collect(x), :, 1)
uniform_pdf = pullback(uniform_map, X)
normal_pdf = pullback(normal_map, X)
pullback_fig, pullback_ax, plt = lines(
    x, uniform_pdf; color = :steelblue, label = "Uniform reference",
    axis = (; xlabel = L"x", ylabel = L"p(x)"),
    figure = (; size = (600, 350)),
)
lines!(pullback_ax, x, normal_pdf; color = :darkorange, label = "Normal reference")
lines!(
    pullback_ax, x, pdf.(distribution, x);
    color = :black, linestyle = :dash, label = "Exact"
)
Legend(pullback_fig[2, 1], pullback_ax; orientation = :horizontal, framevisible = false)

mapping_fig = Figure(size = (600, 600))
for (row, M, reference, name, color, grid) in (
        (1, uniform_map, uniform_reference, "Uniform", :steelblue, range(0.02, 0.98; length = 300)),
        (2, normal_map, normal_reference, "Normal", :darkorange, range(-2.5, 2.5; length = 300)),
    )
    inverse_ax = Axis(mapping_fig[row, 1]; title = "To $name", xlabel = L"x", ylabel = L"z")
    forward_ax = Axis(mapping_fig[row, 2]; title = "From $name", xlabel = L"z", ylabel = L"x")
    mappingplot!(inverse_ax, M, x; direction = :inverse, color, label = "Fitted map")
    lines!(
        inverse_ax, x, quantile.(reference, cdf.(distribution, x));
        color = :black, linestyle = :dash, label = "Exact"
    )
    mappingplot!(forward_ax, M, grid; color)
    lines!(
        forward_ax, grid, quantile.(distribution, cdf.(reference, grid));
        color = :black, linestyle = :dash
    )
end

u = range(0.02, 0.98; length = 300)
cdf_fig = Figure(size = (600, 350))
cdf_ax = Axis(cdf_fig[1, 1]; title = "CDF", xlabel = L"x", ylabel = L"F(x)")
quantile_ax = Axis(cdf_fig[1, 2]; title = "Quantile", xlabel = L"u", ylabel = L"F^{-1}(u)")
for (M, reference, name, color) in (
        (uniform_map, uniform_reference, "Uniform reference", :steelblue),
        (normal_map, normal_reference, "Normal reference", :darkorange),
    )
    estimated_cdf = cdf.(reference, vec(inverse(M, X)))
    reference_quantiles = reshape(quantile.(reference, u), :, 1)
    estimated_quantile = vec(evaluate(M, reference_quantiles))
    lines!(cdf_ax, x, estimated_cdf; color, label = name)
    lines!(quantile_ax, u, estimated_quantile; color)
end
lines!(cdf_ax, x, cdf.(distribution, x); color = :black, linestyle = :dash, label = "Exact")
lines!(quantile_ax, u, quantile.(distribution, u); color = :black, linestyle = :dash)
Legend(cdf_fig[2, 1:2], cdf_ax; orientation = :horizontal, framevisible = false)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
