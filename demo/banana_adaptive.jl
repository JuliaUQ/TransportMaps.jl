using TransportMaps
using Distributions
using LinearAlgebra
using CairoMakie
using LaTeXStrings
set_theme!(transportmap_theme())

banana_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)

num_samples = 500

function generate_banana_samples(n_samples::Int)
    samples = Matrix{Float64}(undef, n_samples, 2)

    count = 0
    while count < n_samples
        x1 = randn() * 2
        x2 = randn() * 3 + x1^2

        if rand() < banana_density([x1, x2]) / 0.4
            count += 1
            samples[count, :] = [x1, x2]
        end
    end

    return samples
end

println("Generating samples from banana distribution...")
target_samples = generate_banana_samples(num_samples)
println("Generated $(size(target_samples, 1)) samples")

L = LinearMap(target_samples)

M, results, selected_terms, selected_folds = optimize_adaptive_transportmap(
    target_samples, [3, 6], 5, L, Softplus(2.0)
)

ind_atm = getmultiindexsets(M.polynomialmap[2])

indices_fig = Figure(size = (600, 400))
indices_ax = Axis(
    indices_fig[1, 1]; aspect = DataAspect(),
    xlabel = L"Multi-index $\alpha_1$", ylabel = L"Multi-index $\alpha_2$",
    xticks = 0:maximum(ind_atm[:, 1]), yticks = 0:maximum(ind_atm[:, 2]),
    limits = (-0.5, maximum(ind_atm[:, 1]) + 0.5, -0.5, maximum(ind_atm[:, 2]) + 0.5)
)
scatter!(indices_ax, ind_atm[:, 1], ind_atm[:, 2]; markersize = 24)

new_samples = generate_banana_samples(1000)
norm_samples = randn(1000, 2)

mapped_banana_samples = inverse(M, norm_samples)

comparison_fig = Figure(size = (600, 480))
comparison_ax = Axis(
    comparison_fig[1, 1];
    xlabel = L"x_1", ylabel = L"x_2", aspect = DataAspect()
)
original_plot = sampleplot!(comparison_ax, new_samples; color = (:steelblue, 0.7))
mapped_plot = sampleplot!(comparison_ax, mapped_banana_samples; color = (:orange, 0.7))
Legend(comparison_fig[1, 2], [original_plot, mapped_plot], ["Original Samples", "Mapped Samples"])

x₁ = range(-3, 3, length = 100)
x₂ = range(-2.5, 5.0, length = 100)

true_density = [banana_density([x1, x2]) for x1 in x₁, x2 in x₂]
learned_density = [pullback(M, [x1, x2]) for x1 in x₁, x2 in x₂]

density_fig = Figure(size = (600, 400))
true_ax = Axis(density_fig[1, 1]; title = "True density", xlabel = L"x_1", ylabel = L"x_2")
learned_ax = Axis(density_fig[1, 2]; title = "Learned density", xlabel = L"x_1", ylabel = L"x_2")
# Use identical contour levels to compare the two densities.
levels = range(0, max(maximum(true_density), maximum(learned_density)); length = 12)[2:(end - 1)]
contour!(true_ax, x₁, x₂, true_density; colormap = :viridis, levels)
contour!(learned_ax, x₁, x₂, learned_density; colormap = :viridis, levels)
linkaxes!(true_ax, learned_ax)

map_index = 2  # Choose 2nd component
best_fold = selected_folds[map_index]
res_best = results[map_index][best_fold]

max_1 = maximum(res_best.terms[end][:, 1])
max_2 = maximum(res_best.terms[end][:, 2])

iterations_fig = Figure(size = (600, 850))
for (i, term) in enumerate(res_best.terms)
    ax = Axis(
        iterations_fig[cld(i, 2), mod1(i, 2)]; title = "Iteration $i",
        aspect = DataAspect(), xlabel = L"Multi-index $\alpha_1$", ylabel = L"Multi-index $\alpha_2$",
        xticks = 0:max_1, yticks = 0:max_2, limits = (-0.5, max_1 + 0.5, -0.5, max_2 + 0.5)
    )
    scatter!(ax, term[:, 1], term[:, 2]; markersize = 20)
end

objectives_fig, objectives_ax, plt = convergenceplot(
    res_best; show_test = true,
    figure = (; size = (600, 400)),
)
axislegend(objectives_ax)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
