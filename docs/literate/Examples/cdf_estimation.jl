# # CDF Estimation with a 1D Transport Map
#
# In this example, we show the use of transport maps to fit (inverse) CDFs [wu2025](@cite).
# We fit a logistic density using uniform and standard-normal references, then
# compare the maps and their PDF, CDF, and quantile estimates.

using TransportMaps, Distributions, CairoMakie, LaTeXStrings
set_theme!(transportmap_theme())

# ## Logistic density
#
# The standard logistic distribution has CDF ``F(x)=1/(1+e^{-x})`` and density
# ``p(x)=F(x)(1-F(x))``. Only the log-density is used for fitting.

distribution = Logistic(0, 1)
x = range(-5, 5; length = 300)
density_fig, density_ax, plt = lines(
    x, pdf.(distribution, x);
    axis = (; xlabel = L"x", ylabel = L"p(x)"),
    figure = (; size = (600, 300)),
)
#md save("logistic-density.svg", density_fig); nothing # hide
# ![Standard logistic density](logistic-density.svg)

# ## Fit both references
#
# Both maps have degree nine and use reference-matched Gaussian quadrature at
# level eight. The uniform map uses shifted Legendre polynomials, and the normal
# map uses the default linearized Hermite basis. Density-based fitting constructs ``T: z \mapsto x``;
# its inverse ``S=T^{-1}`` maps the logistic target into reference space.

target = MapTargetDensity(x -> logpdf(distribution, x[1]))
uniform_reference = Uniform(0, 1)
normal_reference = Normal()
uniform_map = PolynomialMap(1, 9, uniform_reference, Softplus(), ShiftedLegendreBasis())
normal_map = PolynomialMap(1, 9, normal_reference, Softplus(), LinearizedHermiteBasis())
uniform_result = optimize!(uniform_map, target, GaussLegendreWeights(8, uniform_map))
normal_result = optimize!(normal_map, target, GaussHermiteWeights(8, normal_map))
#md nothing # hide

# ## Pullback densities
#
# For either reference density ``\rho``, the fitted target PDF is
# ``\hat p(x)=\rho(S(x))|S'(x)|``. This comparison reveals slope errors that can
# be hard to see in the map curves.

X = reshape(collect(x), :, 1)
uniform_pdf = pullback(uniform_map, X)
normal_pdf = pullback(normal_map, X)
pullback_fig, pullback_ax, plt = lines(
    x, uniform_pdf; color = :steelblue, label = "Uniform reference",
    axis = (; xlabel = L"x", ylabel = L"p(x)"),
    figure = (; size = (600, 350)),
)
lines!(pullback_ax, x, normal_pdf; color = :darkorange, label = "Normal reference")
lines!(
    pullback_ax, x, pdf.(distribution, x);
    color = :black, linestyle = :dash, label = "Exact"
)
Legend(pullback_fig[2, 1], pullback_ax; orientation = :horizontal, framevisible = false)
#md save("logistic-pullback.svg", pullback_fig); nothing # hide
# ![Uniform- and normal-reference pullback PDFs compared with the exact logistic density](logistic-pullback.svg)

# With these reference-matched bases, the normal-reference PDF is smoother and
# closer to the target. This depends on the basis and fit settings as well as
# the reference distribution.

# ## Maps in both directions
#
# If ``G`` is the reference CDF, the exact maps are ``S(x)=G^{-1}(F(x))`` and
# ``T(z)=F^{-1}(G(z))``. For the uniform reference, these reduce to the logistic
# CDF and quantile. The normal-reference maps have different output/input scales.

mapping_fig = Figure(size = (600, 600))
for (row, M, reference, name, color, grid) in (
        (1, uniform_map, uniform_reference, "Uniform", :steelblue, range(0.02, 0.98; length = 300)),
        (2, normal_map, normal_reference, "Normal", :darkorange, range(-2.5, 2.5; length = 300)),
    )
    inverse_ax = Axis(mapping_fig[row, 1]; title = "To $name", xlabel = L"x", ylabel = L"z")
    forward_ax = Axis(mapping_fig[row, 2]; title = "From $name", xlabel = L"z", ylabel = L"x")
    mappingplot!(inverse_ax, M, x; direction = :inverse, color, label = "Fitted map")
    lines!(
        inverse_ax, x, quantile.(reference, cdf.(distribution, x));
        color = :black, linestyle = :dash, label = "Exact"
    )
    mappingplot!(forward_ax, M, grid; color)
    lines!(
        forward_ax, grid, quantile.(distribution, cdf.(reference, grid));
        color = :black, linestyle = :dash
    )
end
#md save("logistic-maps.svg", mapping_fig); nothing # hide
# ![Fitted maps between logistic and uniform or normal distributions; dashed curves show the exact maps](logistic-maps.svg)

# ## CDF and quantile on common scales
#
# To compare the same quantities for both references, use
# ``\hat F(x)=G(S(x))`` and ``\hat F^{-1}(u)=T(G^{-1}(u))``.
# In particular, the normal-reference inverse map is not itself a CDF.
# We omit ``u=0,1``, where the logistic quantile diverges.

u = range(0.02, 0.98; length = 300)
cdf_fig = Figure(size = (600, 350))
cdf_ax = Axis(cdf_fig[1, 1]; title = "CDF", xlabel = L"x", ylabel = L"F(x)")
quantile_ax = Axis(cdf_fig[1, 2]; title = "Quantile", xlabel = L"u", ylabel = L"F^{-1}(u)")
for (M, reference, name, color) in (
        (uniform_map, uniform_reference, "Uniform reference", :steelblue),
        (normal_map, normal_reference, "Normal reference", :darkorange),
    )
    estimated_cdf = cdf.(reference, vec(inverse(M, X)))
    reference_quantiles = reshape(quantile.(reference, u), :, 1)
    estimated_quantile = vec(evaluate(M, reference_quantiles))
    lines!(cdf_ax, x, estimated_cdf; color, label = name)
    lines!(quantile_ax, u, estimated_quantile; color)
end
lines!(cdf_ax, x, cdf.(distribution, x); color = :black, linestyle = :dash, label = "Exact")
lines!(quantile_ax, u, quantile.(distribution, u); color = :black, linestyle = :dash)
Legend(cdf_fig[2, 1:2], cdf_ax; orientation = :horizontal, framevisible = false)
#md save("logistic-cdf.svg", cdf_fig); nothing # hide
# ![Logistic CDF and quantile estimates using uniform and normal references](logistic-cdf.svg)
