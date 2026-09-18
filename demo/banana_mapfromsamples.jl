using TransportMaps
using Distributions
using Optim
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())
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

normal_comparison = reference_target_plot(
    normal_composed_map, validation_samples;
    xgrid = x₁, ygrid = x₂,
    reference_axis = (; limits = ((-4, 4), (-4, 4))),
    figure = (; size = (600, 400)),
)

uniform_comparison = reference_target_plot(
    uniform_composed_map, validation_samples;
    xgrid = x₁, ygrid = x₂,
    reference_axis = (; limits = ((0, 1), (0, 1))),
    figure = (; size = (600, 400)),
)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
