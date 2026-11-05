# # Adaptive Transport Maps
#
# A key challenge in constructing transport maps is choosing the appropriate parameterization,
# specifically the multi-index set that defines which polynomial terms to include in the expansion.
# While fixed parameterizations like total order, no-mixed terms, or diagonal maps (see [Choosing a Map Parameterization](@ref))
# can work well in many cases, they may not be optimal for all target distributions.
#
# Adaptive transport maps address this limitation by automatically selecting the most relevant
# polynomial terms through a greedy enrichment strategy. This approach is particularly useful when:
# - The structure of the target distribution is unknown a priori
# - Computational resources are limited and a sparse representation is desired
# - High-dimensional problems require careful selection of interaction terms

# ## Theory
#
# The adaptive transport map (ATM) algorithm was introduced by [baptista2023](@cite) and provides
# a principled approach to construct sparse, triangular transport maps by adaptively enriching
# the multi-index set based on gradient information.

# ### Greedy Multi-Index Selection
#
# Given a triangular transport map with components $T^k: \mathbb{R}^k \to \mathbb{R}$ for $k=1,\ldots,d$,
# each component is parameterized by a polynomial expansion:
# ```math
# T^k(z_1, \ldots, z_k; \boldsymbol{a}) = f(z_1, \ldots, z_{k-1}, 0; \boldsymbol{a}) + \int_0^{z_k} g\left(\partial_k f(z_1, \ldots, z_{k-1}, \xi; \boldsymbol{a})\right) d\xi
# ```
# where $f$ is a multivariate polynomial:
# ```math
# f(z_1, \ldots, z_k; \boldsymbol{a}) = \sum_{\alpha \in \mathcal{A}_k} a_\alpha \Psi_\alpha(z_1, \ldots, z_k)
# ```
# Here, $\mathcal{A}_k$ is the multi-index set that determines which terms are included.

# The ATM algorithm starts with a minimal multi-index set (typically containing only the constant term)
# and iteratively adds terms that maximize the improvement in the objective function.
# At each iteration $t$, given the current multi-index set $\mathcal{A}_t$, the algorithm:
#
# 1. Identifies candidate terms from the **reduced margin** of $\mathcal{A}_t$:
# ```math
# \mathcal{A}_{\mathrm{RM}}(\mathcal{A}_t) = \{\alpha \notin \mathcal{A}_t : \alpha - e_i \in \mathcal{A}_t \text{ for all } i \text{ with } \alpha_i > 0\}
# ```
# where $e_i$ is the $i$-th standard basis vector.
#
# 2. For each candidate $\alpha \in \mathcal{A}_{\mathrm{RM}}(\mathcal{A}_t)$, evaluates the gradient of the objective
#    with respect to the coefficient $a_\alpha$ (initialized to zero).
#
# 3. Selects the candidate with the largest absolute gradient value:
# ```math
# \alpha^+ = \arg\max_{\alpha \in \mathcal{A}_{\mathrm{RM}}(\mathcal{A}_t)} \left|\frac{\partial J}{\partial a_\alpha}\right|
# ```
#
# 4. Updates the multi-index set: $\mathcal{A}_{t+1} = \mathcal{A}_t \cup \{\alpha^+\}$ and optimizes all coefficients.

# This greedy selection strategy ensures that at each iteration, the term most likely to improve
# the objective function is added, leading to sparse and efficient representations.

# ### Cross-Validation for Model Selection
#
# A critical question when using adaptive transport maps is: how many terms should be included?
# Including too few terms may result in underfitting, while including too many can lead to overfitting.
#
# To address this, the ATM implementation supports **k-fold cross-validation**. The algorithm:
# 1. Splits the data into $k$ folds
# 2. For each fold, trains the map on $k-1$ folds and validates on the remaining fold
# 3. Tracks both training and validation objectives at each iteration
# 4. Selects the number of terms that minimizes the average validation objective across folds

# This approach provides a data-driven way to balance model complexity and generalization performance.

# ## Usage in TransportMaps.jl
#
# The `optimize_adaptive_transportmap` function provides interfaces for constructing adaptive transport maps
# from either samples or a known density function.

# ### Adaptive Maps from Samples
#
# When working with sample data, the simplest approach uses a fixed train-test split to monitor overfitting:
# ```julia
# M, histories = optimize_adaptive_transportmap(samples, maxterms; test_fraction = 0.2)
# ```
#
# For automatic model selection, use k-fold cross-validation. This implementation is based
# on the original algorithm proposed in [baptista2023](@cite):
# ```julia
# M, fold_histories, selected_terms, selected_folds =
#     optimize_adaptive_transportmap(samples, maxterms, k_folds)
# ```

# Here `samples` has one sample per row, and `maxterms` is a vector of term limits
# per component. The defaults use `LinearMap(samples)`, `Softplus()`, and
# `LinearizedHermiteBasis()`. Pass these as additional positional arguments to
# customize them; `optimizer` and `options` are keyword arguments. See
# [`optimize_adaptive_transportmap`](@ref) for the full interface.
#
# The k-fold version returns:
# - `M`: The final composed transport map trained on all data with the selected number of terms
# - `fold_histories`: Optimization histories for each component and fold
# - `selected_terms`: Number of terms selected for each component based on cross-validation
# - `selected_folds`: Which fold had the best performance for each component

# !!! example "Example from Samples"
#     The usage is demonstrated in the example [Banana: Adaptive Map from Samples](@ref).

# ### Adaptive Maps from Density
#
# When the target density function is known analytically, adaptive maps can be constructed directly
# without requiring samples. This approach uses quadrature methods for integration and adaptively
# enriches the multi-index set across all components simultaneously:
# ```julia
# M, history = optimize_adaptive_transportmap(target, quadrature, maxterms;
#     validation = validation_quadrature)
# ```

# `target` wraps a callable log-density; `maxterms` is a single global term budget.
# Omit `validation` when no separate rule is available. The keyword defaults are
# `rectifier=Softplus()`, `basis=LinearizedHermiteBasis()`, and
# `reference_density=Normal()`; `optimizer` and `options` configure the solver.
#
# Key differences from the sample-based approach:
# - Uses a single global budget of terms (`maxterms`) shared across all components
# - Selects which component to enrich at each iteration based on gradient information
# - Supports optional validation using a separate quadrature rule
# - Returns the map with the best validation KL divergence (if validation is provided)

# The returned history contains:
# - `maps`: Array of maps at each iteration
# - `train_objectives`: Training KL divergence values
# - `test_objectives`: Validation KL divergence values (if validation provided)
# - `gradients`: Gradient metrics for all candidates at each iteration

# !!! example "Example from Density"
#     The usage is demonstrated in the example [Cubic: Adaptive Map from Density](@ref).

# ## References
#
# The implementation of adaptive transport maps is based on the work by Baptista et al.:
#
# - Baptista, R., Marzouk, Y., & Zahm, O. (2023). On the Representation and Learning of Monotone Triangular Transport Maps. *Foundations of Computational Mathematics*. <https://doi.org/10.1007/s10208-023-09630-x>
# - Matlab implementation of the original ATM algorithm: <https://github.com/baptistar/ATM>
