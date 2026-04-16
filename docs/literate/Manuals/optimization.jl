
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
# \min_{\boldsymbol{a}} \sum_{i=1}^{N} w_{q,i}\Big[-\log\pi\bigl(T(\boldsymbol{a},\boldsymbol{z}_{q,i})\bigr)-\log |\det\nabla T(\boldsymbol{a},\boldsymbol{z}_{q,i}) |\Big]
# ```

# As noted by [marzouk2016](@cite), this optimization problem is generally non-convex.
# Specifically, it is only convex when the target density $\pi(\boldsymbol{x})$ is log-concave.
# Especially in Bayesian inference, where the target density represents the posterior density, the function is not log-concave, resulting in a non-convex optimization problem.

# In this package, map optimization is performed with the help of [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/), and support a wide range of optimizers and options (such as convergence criteria and printing preferences).
# Specifically, we can pass our `optimize!` function the desired `optimizer` and `options`. For a full overview of available options, see the [Optim.jl configuration documentation](https://julianlsolvers.github.io/Optim.jl/stable/user/config/).

# To perform the optimization of the map coefficients, we call:
# ```julia
# optimize!(M::PolynomialMap, target_density::Function, quadrature::AbstractQuadratureWeights;
#   optimizer::Optim.AbstractOptimizer = LBFGS(), options::Optim.Options = Optim.Options())
# ```

# We have to provide the polynomial map `M`, the target density function, and a quadrature scheme.
# Optionally, we can specify the optimizer (default is `LBFGS()`) and options.

# !!! note "Set initial coefficients"
#     As the starting point of the optimization, the map coefficients can be set using `setcoefficients!(M, coeffs)`, where `coeffs` is a vector of coefficients.

# ###  Usage

# First we load the packages:
using TransportMaps
using Optim
using Distributions
using Plots

# Then, define the target density and quadrature scheme. Here, we use the same banana-shaped density as in [Banana: Map from Density](@ref):
banana_density(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
target = MapTargetDensity(banana_density)
quadrature = GaussHermiteWeights(3, 2)
#md nothing #hide

# Set optimization options to print the trace every 20 iterations:
opts_trace = Optim.Options(iterations=200, show_trace=true, show_every=20, store_trace=true)

# We will try the following optimizers from `Optim.jl`, ordered from simplest to most sophisticated:

# ### Gradient Descent
# The most basic optimization algorithm, Gradient Descent iteratively moves in the direction of the negative gradient. It is simple and robust, but can be slow to converge, especially for ill-conditioned problems.
M_gd = PolynomialMap(2, 2)
res_gd = optimize!(M_gd, target, quadrature; optimizer=GradientDescent(), options=opts_trace)
println(res_gd)

# ### Conjugate Gradient
# Conjugate Gradient improves upon basic gradient descent by using conjugate directions, which can accelerate convergence for large-scale or quadratic problems. It requires gradient information but not the Hessian.
M_cg = PolynomialMap(2, 2)
res_cg = optimize!(M_cg, target, quadrature; optimizer=ConjugateGradient(), options=opts_trace)
println(res_cg)

# ### Nelder-Mead
# Nelder-Mead is a derivative-free optimizer that uses a simplex of points to search for the minimum. It is useful when gradients are unavailable or unreliable, but may be less efficient for high-dimensional or smooth problems.
M_nm = PolynomialMap(2, 2)
res_nm = optimize!(M_nm, target, quadrature; optimizer=NelderMead(), options=opts_trace)
println(res_nm)

# ### BFGS
# BFGS is a quasi-Newton method that builds up an approximation to the Hessian matrix using gradient evaluations. It is generally faster and more robust than gradient descent and conjugate gradient for smooth problems.
M_bfgs = PolynomialMap(2, 2)
res_bfgs = optimize!(M_bfgs, target, quadrature; optimizer=BFGS(), options=opts_trace)
println(res_bfgs)

# ### LBFGS
# LBFGS is a limited-memory version of BFGS, making it suitable for large-scale problems where storing the full Hessian approximation is impractical. It is the default optimizer in many scientific computing packages due to its efficiency and reliability.
M_lbfgs = PolynomialMap(2, 2)
res_lbfgs = optimize!(M_lbfgs, target, quadrature; optimizer=LBFGS(), options=opts_trace)
println(res_lbfgs)

# Finally, we can compare the results by means of variance diagnostic:
samples_z = randn(1000, 2)
v_gd = variance_diagnostic(M_gd, target, samples_z)
v_cg = variance_diagnostic(M_cg, target, samples_z)
v_nm = variance_diagnostic(M_nm, target, samples_z)
v_bfgs = variance_diagnostic(M_bfgs, target, samples_z)
v_lbfgs = variance_diagnostic(M_lbfgs, target, samples_z)

println("Variance diagnostic GradientDescent:   ", v_gd)
println("Variance diagnostic ConjugateGradient: ", v_cg)
println("Variance diagnostic NelderMead:        ", v_nm)
println("Variance diagnostic BFGS:              ", v_bfgs)
println("Variance diagnostic LBFGS:             ", v_lbfgs)

# We can visualize the convergence of all optimizers:
plot([res_gd.trace[i].iteration for i in 1:length(res_gd.trace)], lw=2,
    [res_gd.trace[i].g_norm for i in 1:length(res_gd.trace)], label="GradientDescent")
plot!([res_cg.trace[i].iteration for i in 1:length(res_cg.trace)], lw=2,
    [res_cg.trace[i].g_norm for i in 1:length(res_cg.trace)], label="ConjugateGradient")
plot!([res_nm.trace[i].iteration for i in 1:length(res_nm.trace)], lw=2,
    [res_nm.trace[i].g_norm for i in 1:length(res_nm.trace)], label="NelderMead")
plot!([res_bfgs.trace[i].iteration for i in 1:length(res_bfgs.trace)], lw=2,
    [res_bfgs.trace[i].g_norm for i in 1:length(res_bfgs.trace)], label="BFGS")
plot!([res_lbfgs.trace[i].iteration for i in 1:length(res_lbfgs.trace)], lw=2,
    [res_lbfgs.trace[i].g_norm for i in 1:length(res_lbfgs.trace)], label="LBFGS")
plot!(xaxis=:log, yaxis=:log, xlabel="Iteration", ylabel="Gradient norm",
    title="Convergence of different optimizers", xlims=(1, 200),
    legend=:bottomleft)
#md savefig("optimization-conv.svg"); nothing # hide
# ![Optimization Convergence](optimization-conv.svg)

# It becomes clear, that LBFGS and BFGS are the most efficient optimizers in this case, while Nelder-Mead struggles to keep up.

# ## Map-from-samples

# Another strategy of constructing a transport map is to use samples of the target density, as seen in [Banana: Map from Samples](@ref).
# The formulation of transport map estimation in this way has the benefit to transform the problem into a convex optimization problem, when reference density is log-concave [marzouk2016](@cite).
# Since we can choose the reference density, we can leverage this property to simplify the optimization process.

# When the map is constructed from samples, the optimization problem is formulated by
# minimizing the Kullback-Leibler divergence between the pushforward of the reference density and the empirical distribution of the samples.
# We denote the transport map by $S$, which pushes forward the target distribution to the reference distribution.
# This leads to the following optimization problem:
# ```math
# \min_{\boldsymbol{a}} -\frac{1}{M} \sum_{i=1}^{M} \log \rho\left(S(\boldsymbol{a}, \boldsymbol{x}_i)\right) - \log \left|\det \nabla S(\boldsymbol{a}, \boldsymbol{x}_i)\right|
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
