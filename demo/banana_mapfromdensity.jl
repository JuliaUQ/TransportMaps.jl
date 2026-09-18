using TransportMaps
using Distributions
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())

target_density(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
target = MapTargetDensity(target_density)

normal_map = PolynomialMap(2, 2, Normal(), Softplus())
normal_quadrature = SparseSmolyakWeights(3, normal_map)
normal_result = optimize!(normal_map, target, normal_quadrature)
println(normal_result)

normal_ref_samples = randn(2000, 2)
normal_target_samples = evaluate(normal_map, normal_ref_samples)
var_diag = variance_diagnostic(normal_map, target, normal_ref_samples)

println("Normal-reference KL divergence: ", normal_result.minimum)
println("Normal-reference variance diag: ", var_diag)

uniform_distribution = Uniform(0, 1)
uniform_map = PolynomialMap(
    2,
    5,
    uniform_distribution,
    Softplus(),
    ShiftedLegendreBasis(),
)
uniform_quadrature = SparseSmolyakWeights(3, uniform_map)
uniform_result = optimize!(uniform_map, target, uniform_quadrature)
println(uniform_result)

uniform_ref_samples = rand(uniform_distribution, 2000, 2)
uniform_target_samples = evaluate(uniform_map, uniform_ref_samples)
var_diag_uniform = variance_diagnostic(uniform_map, target, uniform_ref_samples)

println("Uniform-reference KL divergence: ", uniform_result.minimum)
println("Uniform-reference variance diag: ", var_diag_uniform)

x₁ = range(-4, 4, length = 120)
x₂ = range(-3, 7, length = 120)
normal_comparison = reference_target_plot(
    normal_map, normal_ref_samples;
    density = target,
    xgrid = x₁, ygrid = x₂,
    reference_axis = (; limits = ((-4, 4), (-4, 4))),
)

uniform_comparison = reference_target_plot(
    uniform_map, uniform_ref_samples;
    density = target,
    xgrid = x₁, ygrid = x₂,
    reference_axis = (; limits = ((0, 1), (0, 1))),
)

grid = range(-2, 2; length = 100)
mapping_fig = Figure(size = (600, 400))
reference_ax = Axis(mapping_fig[1, 1]; title = "Reference", xlabel = L"z_1", ylabel = L"z_2")
target_ax = Axis(mapping_fig[1, 2]; title = "Mapped grid", xlabel = L"x_1", ylabel = L"x_2")
identity_map = LinearMap(zeros(2), ones(2))
mappingplot!(reference_ax, identity_map, (grid, grid); gridlines = 9)
mappingplot!(target_ax, normal_map, (grid, grid); gridlines = 9)

term_fig = Figure(size = (600, 350))
termplot(
    term_fig[1, 1], normal_map[2];
    axis = (; title = "Coefficients", xlabel = L"a_\alpha")
)
termplot(
    term_fig[1, 2], normal_map[2];
    measure = :removal_rms, samples = normal_ref_samples[1:300, :],
    axis = (; title = "Removal effect", xlabel = "RMS change")
)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
