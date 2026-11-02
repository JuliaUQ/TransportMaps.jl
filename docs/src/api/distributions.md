# Transport Maps as Distributions

```@meta
CurrentModule = TransportMaps
```

[`TransportMapDistribution`](@ref) wraps a fitted transport map as a continuous multivariate distribution, providing the standard `Distributions.jl` methods for sampling, density evaluation, and statistics.

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
