using TransportMaps
using Distributions
using LinearAlgebra
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())

target_density(x) = logpdf(Normal(0, 0.5), x[1]) + logpdf(Normal(0, 0.1), x[2] - x[1]^3)

target = MapTargetDensity(target_density)

quadrature = SparseSmolyakWeights(3, 2)

T, hist = optimize_adaptive_transportmap(
    target, quadrature, 10;
    validation = LatinHypercubeWeights(100, 2)
)
println(hist)

convergence_fig = Figure(size = (600, 650))
objective_ax = Axis(
    convergence_fig[1, 1]; xlabel = "Iteration", yscale = log10,
    ylabel = "KL divergence"
)
convergenceplot!(objective_ax, hist; show_test = true)
axislegend(objective_ax)

grad_norms = [maximum(abs, g; init = 0.0) for g in hist.gradients[2:end]]
gradient_ax = Axis(
    convergence_fig[2, 1]; xlabel = "Iteration", yscale = log10,
    ylabel = "Maximum Absolute Gradient"
)
scatterlines!(gradient_ax, 2:length(hist.gradients), grad_norms)
linkxaxes!(objective_ax, gradient_ax)

samples_z = randn(2000, 2)
mapped_samples = evaluate(T, samples_z)

x1 = -2:0.01:2
x2 = -2:0.01:2

pdf_val = [pdf(target, [x₁, x₂]) for x₁ in x1, x₂ in x2]

density_fig, density_ax, plt = sampleplot(
    mapped_samples; color = (:steelblue, 0.5),
    axis = (; xlabel = L"x_1", ylabel = L"x_2"),
)
contour!(density_ax, x1, x2, pdf_val; colormap = :viridis)

terms_fig = Figure(size = (600, 600))
for k in 1:2
    indices = getmultiindexsets(T[k])
    ys = k == 1 ? zeros(size(indices, 1)) : indices[:, 2]
    ax = Axis(
        terms_fig[k, 1]; title = "Component $k", aspect = DataAspect(),
        xlabel = L"Multi-index $\alpha_1$", ylabel = k == 1 ? "" : L"Multi-index $\alpha_2$",
        xticks = 0:maximum(indices[:, 1]), yticks = 0:maximum(ys),
        limits = (-0.5, maximum(indices[:, 1]) + 0.5, -0.5, maximum(ys) + 0.5)
    )
    scatter!(ax, indices[:, 1], ys; markersize = 20)
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
