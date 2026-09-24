# Run with `julia --project=docs docs/qnorm_animation.jl` to regenerate the asset.
using CairoMakie
using LaTeXStrings
using TransportMaps

set_theme!(transportmap_theme())

p, d = 5, 2
q_range = range(0.1, 1.0; length = 10)
q = Observable(first(q_range))
x = range(0.0, p; length = 1000)

total_points = Point2f.(multivariate_indices(p, d; mode = :total))
hyperbolic_points = @lift(Point2f.(multivariate_indices(p, d; mode = :hyperbolic, q = $q)))
# Guard against small negative roundoff errors before fractional exponentiation.
boundary = @lift(max.(p^$q .- abs.(x) .^ $q, 0.0) .^ (1 / $q))
q_label = lift(q) do value
    L"q = %$(round(value; digits = 2))"
end

fig = Figure(size = (600, 600))
ax = Axis(
    fig[1, 1];
    xlabel = L"Multi-index $\alpha_1$", ylabel = L"Multi-index $\alpha_2$",
    aspect = DataAspect(),
    limits = (-0.5, p + 0.5, -0.5, p + 0.5), xticks = 0:p, yticks = 0:p,
)
total_plot = scatter!(ax, total_points; markersize = 32, color = :steelblue)
hyperbolic_plot = scatter!(ax, hyperbolic_points; markersize = 16, color = :orange)
boundary_plot = lines!(ax, x, boundary; color = :black, linewidth = 2)
axislegend(
    ax, [total_plot, hyperbolic_plot, boundary_plot],
    ["Total Order", q_label, L"\Vert\alpha\Vert_q = p"]; position = :rt
)

record(fig, joinpath(@__DIR__, "src", "assets", "qnorm_animation.gif"), q_range; framerate = 1, px_per_unit = 1) do value
    q[] = value
end
