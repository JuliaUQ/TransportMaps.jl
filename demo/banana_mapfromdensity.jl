using TransportMaps
using Distributions
using Plots

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
target_pdf = [pdf(target, [x1, x2]) for x2 in x₂, x1 in x₁]

function reference_target_plot(
        reference_samples,
        target_samples;
        reference_title,
        reference_limits,
        target_title,
    )
    reference_plot = scatter(
        reference_samples[:, 1],
        reference_samples[:, 2];
        markersize = 2,
        markerstrokewidth = 0,
        alpha = 0.35,
        label = "Reference samples",
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
        markersize = 2,
        markerstrokewidth = 0,
        alpha = 0.45,
        label = "evaluate(M, z)",
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
    normal_ref_samples,
    normal_target_samples;
    reference_title = "Standard-normal reference",
    reference_limits = (-4, 4),
    target_title = "Target from normal reference",
)

uniform_comparison = reference_target_plot(
    uniform_ref_samples,
    uniform_target_samples;
    reference_title = "Uniform reference",
    reference_limits = (0, 1),
    target_title = "Target from uniform reference",
)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
