using TransportMaps
using Distributions
using Optim
using Plots
using Statistics

function generate_banana_samples(number_samples)
    latent_samples = randn(number_samples, 2)
    return hcat(
        latent_samples[:, 1],
        latent_samples[:, 1] .^ 2 .+ latent_samples[:, 2],
    )
end

target_samples = generate_banana_samples(1_000)
linear_map = LinearMap(target_samples)

normal_map = PolynomialMap(2, 2, Normal(), Softplus())
normal_result = optimize!(normal_map, target_samples, linear_map)
normal_composed_map = ComposedMap(linear_map, normal_map)

validation_samples = generate_banana_samples(1_000)
normal_validation_samples = evaluate(normal_composed_map, validation_samples)

println(
    "Mapped normal-reference mean: ",
    vec(mean(normal_validation_samples; dims = 1)),
)
println(
    "Mapped normal-reference std:  ",
    vec(std(normal_validation_samples; dims = 1)),
)

uniform_distribution = Uniform(0, 1)
uniform_map = PolynomialMap(
    2,
    5,
    uniform_distribution,
    Softplus(),
    ShiftedLegendreBasis(),
)
uniform_result = optimize!(
    uniform_map,
    target_samples,
    linear_map;
    options = Optim.Options(iterations = 120, x_abstol = 1.0e-7),
)
uniform_composed_map = ComposedMap(linear_map, uniform_map)

uniform_validation_samples = evaluate(uniform_composed_map, validation_samples)

println(
    "Mapped uniform-reference mean: ",
    vec(mean(uniform_validation_samples; dims = 1)),
)
println(
    "Mapped uniform-reference std:  ",
    vec(std(uniform_validation_samples; dims = 1)),
)

x₁ = range(-4, 4, length = 120)
x₂ = range(-3.5, 7, length = 120)

grid_points = hcat(
    repeat(collect(x₁), inner = 120),
    repeat(collect(x₂), outer = 120),
)

normal_pullback = reshape(
    pullback(normal_composed_map, grid_points),
    120,
    120,
)

uniform_pullback = reshape(
    pullback(uniform_composed_map, grid_points),
    120,
    120,
)

function reference_target_plot(
        reference_samples,
        target_pdf;
        reference_title,
        reference_limits,
        target_title,
    )
    reference_plot = scatter(
        reference_samples[:, 1],
        reference_samples[:, 2];
        markersize = 3,
        markerstrokewidth = 0,
        alpha = 0.6,
        label = "evaluate(C, x)",
        xlabel = "z₁",
        ylabel = "z₂",
        xlims = reference_limits,
        ylims = reference_limits,
        aspect_ratio = 1,
        title = reference_title,
    )

    target_plot = contour(
        x₁,
        x₂,
        target_pdf;
        levels = 8,
        linewidth = 2,
        color = :viridis,
        colorbar = false,
        label = "Target density",
        xlabel = "x₁",
        ylabel = "x₂",
        aspect_ratio = 1,
        title = target_title,
    )
    scatter!(
        target_plot,
        target_samples[:, 1],
        target_samples[:, 2];
        markersize = 3,
        markerstrokewidth = 0,
        alpha = 0.8,
        label = "Validation samples",
    )

    return plot(
        reference_plot,
        target_plot;
        layout = (1, 2),
        size = (950, 430),
        margin = 4 * Plots.mm,
        left_margin = 7 * Plots.mm,
        bottom_margin = 6 * Plots.mm,
    )
end

normal_comparison = reference_target_plot(
    normal_validation_samples,
    normal_pullback;
    reference_title = "Standard-normal reference",
    reference_limits = (-4, 4),
    target_title = "Banana samples",
)

uniform_comparison = reference_target_plot(
    uniform_validation_samples,
    uniform_pullback;
    reference_title = "Uniform reference",
    reference_limits = (0, 1),
    target_title = "Banana samples",
)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
