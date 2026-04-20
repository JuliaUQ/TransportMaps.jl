using TransportMaps
using Plots
using Distributions

function forward_model(t, θ)
    A = 0.4 + (1.2 - 0.4) * cdf(Normal(), θ[1])
    B = 0.01 + (0.31 - 0.01) * cdf(Normal(), θ[2])
    return A * (1 - exp(-B * t))
end

t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1e-3)

s = scatter(t, D, label="Data", xlabel="Time (t)", ylabel="Biochemical Oxygen Demand (D)",
    size=(600, 400), legend=:topleft)
# Plot model output for some parameter values
t_values = range(0, 5, length=100)
for θ₁ in [-0.5, 0, 0.5]
    for θ₂ in [-0.5, 0, 0.5]
        plot!(t_values, [forward_model(ti, [θ₁, θ₂]) for ti in t_values],
            label="(θ₁ = $θ₁, θ₂ = $θ₂)", linestyle=:dash)
    end
end

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

θ₁ = range(-0.5, 1.5, length=100)
θ₂ = range(-0.5, 3, length=100)

posterior_values = [pdf(target, [θ₁, θ₂]) for θ₂ in θ₂, θ₁ in θ₁]

scatter(mapped_samples[:, 1], mapped_samples[:, 2],
    label="Mapped Samples", alpha=0.5, color=1,
    xlabel="θ₁", ylabel="θ₂", title="Posterior Density and Mapped Samples")
contour!(θ₁, θ₂, posterior_values, colormap=:viridis, label="Posterior Density")

posterior_pullback = [pullback(M, [θ₁, θ₂]) for θ₂ in θ₂, θ₁ in θ₁]

contour(θ₁, θ₂, posterior_values ./ maximum(posterior_values);
    levels=5, colormap=:viridis, colorbar=false,
    label="Target", xlabel="θ₁", ylabel="θ₂")
contour!(θ₁, θ₂, posterior_pullback ./ maximum(posterior_pullback);
    levels=5, colormap=:viridis, linestyle=:dash,
    label="Pullback")

θ₁ = 0.
conditional_samples = conditional_sample(M, θ₁, randn(10_000))

θ_range = 0:0.01:2
int_analytical = gaussquadrature(ξ -> pdf(target, [θ₁, ξ]), 1000, -10., 10.)
posterior_conditional(θ₂) = pdf(target, [θ₁, θ₂]) / int_analytical
conditional_analytical = posterior_conditional.(θ_range)

conditional_mapped = conditional_density(M, θ_range, θ₁)

histogram(conditional_samples, bins=50, normalize=:pdf, α=0.5,
    label="Conditional Samples", xlabel="θ₂", ylabel="π(θ₂ | θ₁=$θ₁)")
plot!(θ_range, conditional_analytical, lw=2, label="Analytical Conditional PDF")
plot!(θ_range, conditional_mapped, lw=2, label="TM Conditional PDF")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
