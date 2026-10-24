# Run with `julia --project=docs docs/qnorm_animation.jl` to regenerate the asset.
using CairoMakie
using LaTeXStrings
using TransportMaps
using FFMPEG_jll

include("plotting.jl")

p, d = 5, 2
q_range = range(0.1, 1.0; length = 10)
x = range(0.0, p; length = 1000)

function qnorm_figure()
    q = Observable(first(q_range))
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
    boundary_plot = lines!(ax, x, boundary; color = theme(:textcolor), linewidth = 2)
    axislegend(
        ax, [total_plot, hyperbolic_plot, boundary_plot],
        ["Total Order", q_label, L"\Vert\alpha\Vert_q = p"]; position = :rt
    )
    return fig, q
end

for variant in DocPlots.VARIANTS
    with_theme(DocPlots.plot_theme(variant)) do
        fig, q = qnorm_figure()
        path = joinpath(@__DIR__, "src", "assets", "qnorm_animation-$variant.png")
        # APNG preserves smooth text and translucent grids, unlike GIF's binary alpha.
        # Render at 2× density and display at 600 pixels in the documentation.
        mktempdir() do frames
            for (i, value) in enumerate(q_range)
                q[] = value
                frame = joinpath(frames, "frame-$(lpad(i, 4, '0')).png")
                save(frame, fig; px_per_unit = 2)
                # Cairo's native PNG writer correctly unpremultiplies translucent pixels.
                screen = CairoMakie.Makie.getscreen(fig.scene)
                CairoMakie.Cairo.write_to_png(screen.surface, frame)
            end
            input = joinpath(frames, "frame-%04d.png")
            run(`$(ffmpeg()) -v error -y -framerate 1 -i $input -plays 0 -f apng $path`)
        end
    end
end
