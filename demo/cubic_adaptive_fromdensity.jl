using TransportMaps
using Distributions
using LinearAlgebra
using Plots

target_density(x) = logpdf(Normal(0, 0.5), x[1]) + logpdf(Normal(0, 0.1), x[2] - x[1]^3)

target = MapTargetDensity(target_density)

quadrature = SparseSmolyakWeights(3, 2)

T, hist = optimize_adaptive_transportmap(
    target, quadrature, 10;
    validation = LatinHypercubeWeights(100, 2)
)
println(hist)

convergence_kl = plot(
    hist.train_objectives, label = "Train Objective", xlabel = "Iteration",
    ylabel = "KL divergence", title = "Objective Value vs Iteration", marker = :o
)
plot!(convergence_kl, hist.test_objectives, label = "Test Objective", marker = :o)
yaxis!(:log10)

grad_norms = maximum.(hist.gradients[2:end])

convergence_grad = plot(
    2:length(hist.gradients), grad_norms; xlabel = "Iteration",
    ylabel = "Maximum Gradient", label = nothing, marker = :o, title = "Gradient"
)
yaxis!(:log10)
xlims!(xlims(convergence_kl))

plot(convergence_kl, convergence_grad, layout = (2, 1))

samples_z = randn(2000, 2)
mapped_samples = evaluate(T, samples_z)

x1 = -2:0.01:2
x2 = -2:0.01:2

pdf_val = [pdf(target, [x₁, x₂]) for x₂ in x2, x₁ in x1]

s = scatter(
    mapped_samples[:, 1], mapped_samples[:, 2],
    label = "Mapped Samples", alpha = 0.5, color = 1,
    xlabel = "x₁", ylabel = "x₂", title = "Target Density and Mapped Samples"
)

contour!(x1, x2, pdf_val)

ind_atm = getmultiindexsets(T[1])
MIS1 = scatter(ind_atm[:, 1], zeros(length(ind_atm)), ms = 20, legend = false)
plot!(
    xlims = (-0.5, maximum(ind_atm[:, 1]) + 0.5), ylims = (-0.5, 0.5),
    aspect_ratio = 1, xlabel = "Multi-index α₁", ylabel = "", title = "Multi-indices Component 1"
)
xticks!(0:maximum(ind_atm[:, 1]))
yticks!(0:0)

ind_atm = getmultiindexsets(T[2])
MIS2 = scatter(ind_atm[:, 1], ind_atm[:, 2], ms = 20, legend = false)
plot!(
    xlims = (-0.5, maximum(ind_atm[:, 1]) + 0.5), ylims = (-0.5, maximum(ind_atm[:, 2]) + 0.5),
    aspect_ratio = 1, xlabel = "Multi-index α₁", ylabel = "Multi-index α₂",
    title = "Multi-indices Component 2"
)
xticks!(0:maximum(ind_atm[:, 1]))
yticks!(0:maximum(ind_atm[:, 2]))

plot(MIS1, MIS2, layout = (2, 1), size = (600, 600))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
