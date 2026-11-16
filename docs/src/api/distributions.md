# Transport Maps as Distributions

```@meta
CurrentModule = TransportMaps
```

[`TransportMapDistribution`](@ref) wraps a fitted transport map as a continuous multivariate distribution, providing the standard `Distributions.jl` methods for sampling, density evaluation, and statistics.

General [`ComposedMap`](@ref) chains can contain any number of polynomial,
linear, Laplace, marginal, or custom invertible layers. The wrapper uses the
composition's reference and direction, which are fixed at construction; rebuild
the composition to change these settings. Layer parameters remain shared.

```julia
prior = MarginalMap([Uniform(-2, 4), LogNormal(0.2, 0.7)])
map = ComposedMap(fitted_correction, prior; forwarddirection = :target)
d = TransportMapDistribution(map)
```

Custom layers must implement `numberdimensions`, `evaluate`, `inverse`, and
`logabsdetjacobian` (or `jacobian`). Batch sampling and statistics also require
the row-wise matrix methods of `evaluate` and `inverse`.
References must be continuous distributions: a `MapReferenceDensity` defines a
product of identical marginals, and a multivariate distribution defines the
joint reference. Density-only `MapTargetDensity` references cannot be wrapped.

For general usage and conventions, see the [Distributions.jl multivariate distribution interface](https://juliastats.org/Distributions.jl/stable/multivariate/#Common-Interface).

## Index

```@index
Pages = ["distributions.md"]
```

## Types

```@docs
TransportMapDistribution
```

## Sampling

Wrap a fitted map using [`TransportMapDistribution`](@ref),
then sample through the standard Distributions.jl interface:

```julia
d = TransportMapDistribution(fitted_map)

x = rand(d)        # One sample
X = rand(d, 1000)  # 1000 samples
```

!!! note "Sample orientation"
    Each column of `X` is one sample. Map-level batch operations
    such as `evaluate` use rows instead.

## Density evaluation

Evaluate densities with `pdf` or `logpdf`, and check support
with `insupport`.

```@docs
pdf(::TransportMapDistribution, ::AbstractVector{<:Real})
logpdf(::TransportMapDistribution, ::AbstractVector{<:Real})
insupport
```

## Moments and entropy

Polynomial and composed maps use reference-space quadrature.
Pass `quad` to choose a different integration rule.
Automatic quadrature supports Normal and Uniform product references and
multivariate normal references, including their correlations. For other
references, pass a rule with points and weights in reference coordinates, such
as `quad = MonteCarloWeights(10_000, map)`. Entropy additionally requires the
reference distribution's `entropy` method.

!!! tip "Exact statistics"
    Standalone `LinearMap` and `LaplaceMap` distributions use
    exact Gaussian formulas and ignore `quad`.

```@docs
mean(d::TransportMapDistribution)
var(d::TransportMapDistribution)
std(d::TransportMapDistribution)
cov(d::TransportMapDistribution)
cor(d::TransportMapDistribution)
entropy(d::TransportMapDistribution)
```
