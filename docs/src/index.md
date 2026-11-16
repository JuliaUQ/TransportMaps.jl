```@raw html
---
layout: home
description: Triangular transport maps for sampling, density estimation, and Bayesian inference in Julia.

hero:
  name: TransportMaps.jl
  text: Transform probability distributions.
  tagline: Build and adapt triangular transport maps for sampling, density estimation, and Bayesian inference in Julia.
  image:
    src: /logo.svg
    alt: TransportMaps.jl logo
  actions:
    - theme: alt
      text: Introduction
      link: /Manuals/introduction
    - theme: brand
      text: Get Started
      link: /Manuals/getting_started
    - theme: alt
      text: API Reference
      link: /api/maps
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaUQ/TransportMaps.jl

features:
  - title: Learn from samples
    details: Fit a transport map directly from observations, with optional linear standardization.
    link: /Examples/banana_mapfromsamples
    linkText: Fit a map from samples
  - title: Start from a density
    details: Construct a map from an unnormalized log-density using quadrature and numerical optimization.
    link: /Examples/banana_mapfromdensity
    linkText: Fit a map from a density
  - title: Distributions.jl interface
    details: Sample, evaluate densities, and compute moments through the standard Distributions.jl interface.
    link: /api/distributions
    linkText: Use a fitted distribution
  - title: Adapt the approximation
    details: Select polynomial terms adaptively and use validation data to choose the map's complexity.
    link: /Manuals/adaptive_transport_map
    linkText: Explore adaptive maps
  - title: Choose your basis and quadrature
    details: Combine polynomial bases, monotone rectifiers, and tensor-product or sparse-grid integration rules.
    link: /Manuals/map_parameterization
    linkText: Configure a transport map
  - title: Inspect the results
    details: Visualize transformations, reference-space diagnostics, pairwise correlations, and optimization histories with Makie.
    link: /Manuals/plotting
    linkText: Plot and diagnose a map

authors:
  - name: Lukas Fritsch
    platform: github
    link: https://github.com/lukasfritsch
  - name: Jan Grashorn
    platform: github
    link: https://github.com/jgrashorn
---
```

```@meta
CurrentModule = TransportMaps
```

## Authors

```@raw html
<Authors />
```

## Install TransportMaps.jl

```julia
using Pkg
Pkg.add("TransportMaps")
```

Read the [Introduction](Manuals/introduction.md) for the mathematical background.
Follow the [Getting Started guide](@ref "Getting Started with TransportMaps.jl") for a complete walkthrough,
or explore [CDF estimation](Examples/cdf_estimation.md) and
[Bayesian inference with BOD data](Examples/bod_bayesianinference.md).

## Mathematical background

Start with the [Introduction](Manuals/introduction.md) for
the change-of-variables formula, triangular maps, and the density- and sample-based
fitting objectives.

Transport maps are smooth, invertible transformations between probability distributions.
Triangular maps build on the Rosenblatt transformation [rosenblatt1952](@cite) and the
Knothe–Rosenblatt rearrangement [knothe1957](@cite), and support sampling, density estimation,
and variational inference [marzouk2016](@cite).

This package implements monotone triangular transport maps following [marzouk2016](@cite)
and [baptista2023](@cite). For an introduction, see [ramgraber2025](@cite);
for applications in structural health monitoring and Bayesian inference, see [grashorn2024](@cite).

The manuals cover [polynomial bases](Manuals/basis_functions.md),
[quadrature rules](Manuals/quadrature_methods.md), and [optimization](Manuals/optimization.md),
including automatic differentiation through `DifferentiationInterface.jl` and multithreaded map evaluation.

[`TransportMapDistribution`](@ref) wraps a fitted map as a continuous multivariate
distribution. Use the same interface whether the map was fitted from a density or
from samples, including compositions with linear or Laplace maps.
The distribution represents the fitted approximation to the target.
The [getting-started guide](Manuals/getting_started.md) demonstrates fitting and
using a distribution; the [distribution API](api/distributions.md) documents
sampling, density evaluation, support checks, moments, and entropy.

## Related implementations

- [ATM](https://github.com/baptistar/ATM): Matlab code for adaptive transport maps [baptista2023](@cite)
- [MParT](https://github.com/MeasureTransport/MParT): C++-based library for transport maps [parno2022](@cite)
- [TransportBasedInference.jl](https://github.com/mleprovost/TransportBasedInference.jl): Julia implementation of adaptive transport maps (ATM) and Kalman filters
- [SequentialMeasureTransport.jl](https://github.com/benjione/SequentialMeasureTransport.jl): Julia implementation of transport maps from sum-of-squares densities [zanger2024](@cite)
- [Triangular-Transport-Toolbox](https://github.com/MaxRamgraber/Triangular-Transport-Toolbox): Python code for the triangular transport tutorial paper [ramgraber2025](@cite)
