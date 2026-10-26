# # Optimization of the Map Coefficients
#
# A crucial step in constructing transport maps is the optimization of the map coefficients, which determine how well the map represents the target distribution.
# This process can be approached in two distinct ways, depending on the available information about the target distribution [marzouk2016](@cite).

# ## Map-from-density

# One way to construct a transport map is to directly optimize its parameters based on the (unnormalized) target density, as shown in [Banana: Map from Density](@ref).
# This approach requires access to the target density function and uses quadrature schemes to approximate integrals, as introduced in [Quadrature Methods](@ref).
#
# Formally, we define the following optimization problem to determine the coefficients $\boldsymbol{a}$ of the parameterized map $T$:
# ```math
# \min_{\boldsymbol{a}} \sum_{i=1}^{N} w_{q,i}\Big[-\log\widetilde{\pi}\bigl(T(\boldsymbol{a},\boldsymbol{z}_{q,i})\bigr)-\log |\det\nabla T(\boldsymbol{a},\boldsymbol{z}_{q,i}) |\Big]
# + \lambda_1 \sum_{j \in P}\sqrt{a_j^2 + \varepsilon^2}
# + \frac{\lambda_2}{2}\sum_{j \in P}a_j^2.
# ```
# Here, ``\widetilde{\pi}`` is the unnormalized target density, as in the
# [Introduction](introduction.md). ``P`` contains all nonlinear coefficients by default, so constant and linear
# terms remain unpenalized. With `interactions_only=true`, ``P`` instead contains only
# terms involving at least two coordinates. The L1 term uses the differentiable
# approximation ``\sqrt{a_j^2 + \varepsilon^2}``, controlled by `l1_eps`.

# As noted by [marzouk2016](@cite), this optimization problem is generally non-convex.
# Log-concavity of the target yields convexity in suitable spaces of maps, but does
# not by itself guarantee convexity in the coefficients of a nonlinearly rectified
# polynomial map. Bayesian posteriors may be log-concave or non-log-concave. See
# [baptista2023](@cite) for the role of the map representation in optimization.

# Map optimization uses [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/),
# which supports a range of optimizers and options, including convergence criteria and trace output.
# Specifically, we can pass our `optimize!` function the desired `optimizer` and `options`. For a full overview of available options, see the [Optim.jl configuration documentation](https://julianlsolvers.github.io/Optim.jl/stable/user/config/).

# To perform the optimization of the map coefficients, we call:
# ```julia
# optimize!(M::PolynomialMap, target::AbstractMapDensity, quadrature::AbstractQuadratureWeights;
#   optimizer::Optim.AbstractOptimizer = LBFGS(), options::Optim.Options = Optim.Options(),
#   λ1::Real = 0, λ2::Real = 0, l1_eps::Real = 1e-8,
#   interactions_only::Bool = false)
# ```

# Provide the polynomial map `M`, a `MapTargetDensity` wrapping the target log-density,
# and a quadrature scheme for the reference distribution.
# Optionally, we can specify the optimizer, its options, and L1 or L2 regularization.
# For example, `λ1=1e-3` shrinks every nonlinear term, while combining it with
# `interactions_only=true` shrinks only mixed-coordinate terms. These keywords are
# also available for density-based `optimize_adaptive_transportmap` calls.

# !!! note "Set initial coefficients"
#     As the starting point of the optimization, the map coefficients can be set using `setcoefficients!(M, coeffs)`, where `coeffs` is a vector of coefficients.

# ###  Usage

# First we load the packages:
using TransportMaps
using Optim
using Distributions
using CairoMakie # hide
using LaTeXStrings # hide
set_theme!(transportmap_theme()) # hide
#md using Main.DocPlots: save_docplot # hide

# Then, define the target density and quadrature scheme. Here, we use the same banana-shaped density as in [Banana: Map from Density](@ref):
logtarget(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
target = MapTargetDensity(logtarget)
M_gd = PolynomialMap(2, 2)
quadrature = GaussHermiteWeights(3, M_gd)
#md nothing #hide

# Store the trace for the convergence comparison below:
opts_trace = Optim.Options(iterations = 200, store_trace = true)
#md nothing # hide

# We will try the following optimizers from `Optim.jl`, ordered from simplest to most sophisticated:

# ### Gradient Descent
# The most basic optimization algorithm, Gradient Descent iteratively moves in the direction of the negative gradient. It is simple and robust, but can be slow to converge, especially for ill-conditioned problems.
res_gd = optimize!(M_gd, target, quadrature; optimizer = GradientDescent(), options = opts_trace)
#md nothing # hide

# ### Conjugate Gradient
# Conjugate Gradient improves upon basic gradient descent by using conjugate directions, which can accelerate convergence for large-scale or quadratic problems. It requires gradient information but not the Hessian.
M_cg = PolynomialMap(2, 2)
res_cg = optimize!(M_cg, target, quadrature; optimizer = ConjugateGradient(), options = opts_trace)
#md nothing # hide

# ### Nelder-Mead
# Nelder-Mead is a derivative-free optimizer that uses a simplex of points to search for the minimum. It is useful when gradients are unavailable or unreliable, but may be less efficient for high-dimensional or smooth problems.
M_nm = PolynomialMap(2, 2)
res_nm = optimize!(M_nm, target, quadrature; optimizer = NelderMead(), options = opts_trace)
#md nothing # hide

# ### BFGS
# BFGS is a quasi-Newton method that builds up an approximation to the Hessian matrix using gradient evaluations. It is generally faster and more robust than gradient descent and conjugate gradient for smooth problems.
M_bfgs = PolynomialMap(2, 2)
res_bfgs = optimize!(M_bfgs, target, quadrature; optimizer = BFGS(), options = opts_trace)
#md nothing # hide

# ### LBFGS
# LBFGS is a limited-memory version of BFGS, making it suitable for large-scale problems where storing the full Hessian approximation is impractical. It is the default optimizer in many scientific computing packages due to its efficiency and reliability.
M_lbfgs = PolynomialMap(2, 2)
res_lbfgs = optimize!(M_lbfgs, target, quadrature; optimizer = LBFGS(), options = opts_trace)
#md nothing # hide

# Compare map quality on the same reference samples. The other optimizer results
# use the same diagnostic:
samples_z = randn(1000, 2)
v_lbfgs = variance_diagnostic(M_lbfgs, target, samples_z)

# We can visualize the convergence of all optimizers:
#md save_docplot("optimization-conv.svg") do # hide
fig = Figure(size = (600, 450)) # hide
ax = Axis( # hide
    fig[1, 1]; xscale = log10, yscale = log10, # hide
    xlabel = "Iteration", ylabel = "Gradient norm" # hide
) # hide
for (result, label) in ( # hide
        (res_gd, "GradientDescent"), (res_cg, "ConjugateGradient"), # hide
        (res_nm, "NelderMead"), (res_bfgs, "BFGS"), (res_lbfgs, "LBFGS"), # hide
    ) # hide
    ## Logarithmic axes require positive, finite coordinates. # hide
    trace = filter(t -> t.iteration > 0 && isfinite(t.g_norm) && t.g_norm > 0, result.trace) # hide
    lines!(ax, [t.iteration for t in trace], [t.g_norm for t in trace]; linewidth = 2, label) # hide
end # hide
xlims!(ax, 1, 200) # hide
axislegend(ax; position = :lb) # hide
#md     fig # hide
#md end # hide
# ![Optimization Convergence](optimization-conv.svg)

# It becomes clear, that LBFGS and BFGS are the most efficient optimizers in this case, while Nelder-Mead struggles to keep up.

# ## Map-from-samples

# Another strategy of constructing a transport map is to use samples of the target density, as seen in [Banana: Map from Samples](@ref).
# A log-concave reference gives a convex objective in the monotone map itself
# [marzouk2016](@cite). The integrated rectification used here depends nonlinearly
# on the coefficients, so this does not establish convexity of the finite-dimensional
# coefficient problem [baptista2023](@cite).

# When the map is constructed from samples, the optimization problem is formulated by
# minimizing ``\mathcal{D}_{\mathrm{KL}}(S_{\#}\pi\,\|\,\rho)`` and approximating the
# target expectation with a sample average, as derived in the [Introduction](introduction.md).
# We denote the transport map by $S$, which pushes forward the target distribution to the reference distribution.
# This leads to the following optimization problem:
# ```math
# \min_{\boldsymbol{a}} -\frac{1}{M} \sum_{i=1}^{M} \left[\log \rho\left(S(\boldsymbol{a}, \boldsymbol{x}_i)\right) + \log \left|\det \nabla S(\boldsymbol{a}, \boldsymbol{x}_i)\right|\right]
# ```
# where $\{\boldsymbol{x}_i\}_{i=1}^M$ are samples from the target distribution, and $\rho(\cdot)$ is the density of the reference distribution.

# To perform the optimization, we can use the same `optimize!` function as before, but now
# we pass samples instead of a target density and quadrature scheme.
# Similarly, we can specify the optimizer and options:
#
# ```julia
# optimize!(M::PolynomialMap, samples::AbstractArray{<:Real};
#   optimizer::Optim.AbstractOptimizer = LBFGS(), options::Optim.Options = Optim.Options())
# ```
