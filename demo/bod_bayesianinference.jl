using TransportMaps
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())
using Distributions

function forward_model(t, parameters)
    A, B = parameters
    return A * (1 - exp(-B * t))
end

prior_map = PriorMap([Uniform(0.4, 1.2), Uniform(0.01, 0.31)])

t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1.0e-3)

realizations_fig = Figure(size = (600, 650))
realizations_ax = Axis(realizations_fig[1, 1]; xlabel = L"Time $t$", ylabel = L"Biochemical Oxygen Demand $D$")
scatter!(realizations_ax, t, D; label = "Data", color = theme(:textcolor))
t_values = range(0, 5; length = 100)
for A in [0.6, 0.8, 1.0], B in [0.1, 0.16, 0.22]
    lines!(
        realizations_ax, t_values, [forward_model(ti, [A, B]) for ti in t_values];
        label = L"(A = %$A,\; B = %$B)", linestyle = :dash
    )
end
Legend(realizations_fig[2, 1], realizations_ax; orientation = :horizontal, nbanks = 5)

loglikelihood(parameters) = sum(logpdf(Normal(forward_model(t[k], parameters), σ), D[k]) for k in eachindex(t))
logposterior(parameters) = loglikelihood(parameters) + logpdf(prior_map, parameters)
physical_target = MapTargetDensity(parameters -> logposterior(parameters))

target = MapTargetDensity(θ -> loglikelihood(evaluate(prior_map, θ)) + logpdf(prior_map.reference, θ))

M = PolynomialMap(2, 3, Normal(), Softplus(), LinearizedHermiteBasis())
quadrature = GaussHermiteWeights(3, M)
res = optimize!(M, target, quadrature)
println("Optimization result: ", res)
posterior_map = ComposedMap(M, prior_map; forwarddirection = :target)

samples_z = randn(1000, 2)
mapped_samples = evaluate(posterior_map, samples_z)

var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)

A_values = range(0.4, 1.2; length = 100)
B_values = range(0.01, 0.31; length = 100)
posterior_values = [pdf(physical_target, [A, B]) for A in A_values, B in B_values]

samples_fig, samples_ax, plt = sampleplot(
    mapped_samples; color = (:steelblue, 0.5),
    axis = (; xlabel = L"A", ylabel = L"B"),
)
limits!(samples_ax, 0.4, 1.2, 0.01, 0.31)
contour!(
    samples_ax, A_values, B_values, posterior_values ./ maximum(posterior_values);
    levels = [0.2, 0.4, 0.6, 0.8], colormap = :viridis
)

posterior_pullback = [pdf(posterior_map, [A, B]) for A in A_values, B in B_values]

pullback_fig = Figure(size = (600, 420))
pullback_ax = Axis(pullback_fig[1, 1]; xlabel = L"A", ylabel = L"B")
levels = 0.2:0.2:0.8
contour!(
    pullback_ax, A_values, B_values, posterior_values ./ maximum(posterior_values);
    levels, color = :steelblue, label = "Target"
)
contour!(
    pullback_ax, A_values, B_values, posterior_pullback ./ maximum(posterior_pullback);
    levels, color = :orange, linestyle = :dash, label = "Transport approximation"
)
limits!(pullback_ax, 0.4, 1.2, 0.01, 0.31)
axislegend(
    pullback_ax, [LineElement(color = :steelblue), LineElement(color = :orange, linestyle = :dash)],
    ["Target", "Transport"]; position = :lb
)

fixed_A = 0.8
fixed_θ₁ = inverse(prior_map, [fixed_A, mean(prior_map.marginals[2])])[1]
conditional_θ₂ = conditional_sample(M, fixed_θ₁, randn(10_000))
conditional_samples = evaluate(prior_map, hcat(fill(fixed_θ₁, length(conditional_θ₂)), conditional_θ₂))[:, 2]

B_range = range(0.011, 0.309; length = 300)
conditional_normalizer = gaussquadrature(B -> pdf(physical_target, [fixed_A, B]), 200, 0.01, 0.31)
reference_conditional(B) = pdf(physical_target, [fixed_A, B]) / conditional_normalizer
conditional_reference = reference_conditional.(B_range)

B_prior_map = PriorMap([prior_map.marginals[2]])
B_points = reshape(collect(B_range), :, 1)
θ₂_range = inverse(B_prior_map, B_points)[:, 1]
conditional_mapped = conditional_density(M, θ₂_range, fixed_θ₁) .* inverse_jacobian(B_prior_map, B_points)

conditional_fig, conditional_ax, plt = hist(
    conditional_samples; bins = 50, normalization = :pdf, color = (:steelblue, 0.5),
    label = "Samples", axis = (; xlabel = L"B", ylabel = L"\pi(B \mid A = %$fixed_A)"),
)
lines!(conditional_ax, B_range, conditional_reference; linewidth = 2, color = :orange, label = "Reference")
lines!(conditional_ax, B_range, conditional_mapped; linewidth = 2, color = :seagreen, label = "Transport")
xlims!(conditional_ax, 0.01, 0.31)
axislegend(conditional_ax; position = :lt)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
