using TransportMaps
using Distributions
using Statistics
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())

function generate_banana_samples(number_samples)
    latent_samples = randn(number_samples, 2)
    return hcat(
        latent_samples[:, 1],
        latent_samples[:, 1] .^ 3 .+ latent_samples[:, 2],
    )
end

target_samples = generate_banana_samples(1_000)

d_samples_adaptive = TransportMapDistribution(
    target_samples,
    [10, 10],  # Maximum polynomial order per dimension
    true;      # Use adaptive term selection
    use_linear_map = true,  # Standardize data first
    k_folds = 5,  # Cross-validation folds
)

println("Samples-based distribution created")
println("  Mean: ", mean(d_samples_adaptive))
println("  Std:  ", std(d_samples_adaptive))

logtarget(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^3)
quadr = SparseSmolyakWeights(3, 2)

d_density_adaptive = TransportMapDistribution(
    logtarget,
    quadr,
    27,    # Target number of polynomial terms
    true  # Use adaptive term selection
)

println("\nDensity-based distribution created")
println("  Mean: ", mean(d_density_adaptive))
println("  Std:  ", std(d_density_adaptive))

samples_from_fit = rand(d_samples_adaptive, 1000)
println("\nGenerated 1000 samples from fitted distribution")
println("  Empirical mean: ", vec(mean(samples_from_fit; dims = 1)))
println("  Empirical std:  ", vec(std(samples_from_fit; dims = 1)))

test_point = [0.5, 0.2]
log_dens_samples = logpdf(d_samples_adaptive, test_point)
log_dens_target = logpdf(d_density_adaptive, test_point)
dens_samples = pdf(d_samples_adaptive, test_point)
dens_target = pdf(d_density_adaptive, test_point)

println("\nDensity evaluation at $test_point:")
println("  From samples: pdf=$(dens_samples), logpdf=$(log_dens_samples)")
println("  From target:  pdf=$(dens_target), logpdf=$(log_dens_target)")

true_log_density = logtarget(test_point)
println("\n  True logtarget: $true_log_density")

println("\nStatistics from samples-based distribution:")
println("  Covariance matrix:\n", cov(d_samples_adaptive))
println("  Entropy: ", entropy(d_samples_adaptive))

println("\nStatistics from density-based distribution:")
println("  Covariance matrix:\n", cov(d_density_adaptive))
println("  Entropy: ", entropy(d_density_adaptive))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
