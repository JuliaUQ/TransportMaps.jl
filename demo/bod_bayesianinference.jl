using TransportMaps
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())
using Distributions

function forward_model(t, θ)
    A = 0.4 + (1.2 - 0.4) * cdf(Normal(), θ[1])
    B = 0.01 + (0.31 - 0.01) * cdf(Normal(), θ[2])
    return A * (1 - exp(-B * t))
end

t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1.0e-3)

realizations_fig = Figure(size = (600, 650))
realizations_ax = Axis(realizations_fig[1, 1]; xlabel = L"Time $t$", ylabel = L"Biochemical Oxygen Demand $D$")
scatter!(realizations_ax, t, D; label = "Data", color = :black)
# Plot model output for some parameter values.
t_values = range(0, 5, length = 100)
for θ₁ in [-0.5, 0, 0.5]
    for θ₂ in [-0.5, 0, 0.5]
        lines!(
            realizations_ax, t_values, [forward_model(ti, [θ₁, θ₂]) for ti in t_values];
            label = L"(\theta_1 = %$θ₁,\; \theta_2 = %$θ₂)", linestyle = :dash
        )
    end
end
Legend(realizations_fig[2, 1], realizations_ax; orientation = :horizontal, nbanks = 5)

function logposterior(θ)
    # Calculate the likelihood
    likelihood = sum([logpdf(Normal(forward_model(t[k], θ), σ), D[k]) for k in 1:5])
    # Calculate the prior
    prior = logpdf(Normal(), θ[1]) + logpdf(Normal(), θ[2])
    return prior + likelihood
end

target = MapTargetDensity(x -> logposterior(x))

M = PolynomialMap(2, 3, :normal, Softplus(), LinearizedHermiteBasis())

quadrature = GaussHermiteWeights(3, 2)

res = optimize!(M, target, quadrature)
println("Optimization result: ", res)

samples_z = randn(1000, 2)

mapped_samples = evaluate(M, samples_z)

var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)

θ₁ = range(-0.5, 1.5, length = 100)
θ₂ = range(-0.5, 3, length = 100)

posterior_values = [pdf(target, [θ₁, θ₂]) for θ₁ in θ₁, θ₂ in θ₂]

samples_fig, samples_ax, plt = sampleplot(
    mapped_samples; color = (:steelblue, 0.5),
    axis = (; xlabel = L"\theta_1", ylabel = L"\theta_2"),
)
contour!(samples_ax, θ₁, θ₂, posterior_values; colormap = :viridis)

posterior_pullback = [pullback(M, [θ₁, θ₂]) for θ₁ in θ₁, θ₂ in θ₂]

pullback_fig = Figure(size = (600, 420))
pullback_ax = Axis(pullback_fig[1, 1]; xlabel = L"\theta_1", ylabel = L"\theta_2")
levels = 0.2:0.2:0.8
contour!(
    pullback_ax, θ₁, θ₂, posterior_values ./ maximum(posterior_values);
    levels, color = :steelblue, label = "Target"
)
contour!(
    pullback_ax, θ₁, θ₂, posterior_pullback ./ maximum(posterior_pullback);
    levels, color = :orange, linestyle = :dash, label = "Pullback"
)
axislegend(pullback_ax)

θ₁ = 0.0
conditional_samples = conditional_sample(M, θ₁, randn(10_000))

θ_range = 0:0.01:2
int_analytical = gaussquadrature(ξ -> pdf(target, [θ₁, ξ]), 1000, -10.0, 10.0)
posterior_conditional(θ₂) = pdf(target, [θ₁, θ₂]) / int_analytical
conditional_analytical = posterior_conditional.(θ_range)

conditional_mapped = conditional_density(M, θ_range, θ₁)

conditional_fig, conditional_ax, plt = hist(
    conditional_samples; bins = 50, normalization = :pdf, color = (:steelblue, 0.5),
    label = "Conditional Samples", axis = (; xlabel = L"\theta_2", ylabel = L"\pi(\theta_2 \mid \theta_1 = %$θ₁)"),
)
lines!(conditional_ax, θ_range, conditional_analytical; linewidth = 2, color = :orange, label = "Analytical Conditional PDF")
lines!(conditional_ax, θ_range, conditional_mapped; linewidth = 2, color = :seagreen, label = "TM Conditional PDF")
axislegend(conditional_ax)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
