```@raw html
---
authors:
  - name: Lukas Fritsch
    platform: github
    link: https://github.com/lukasfritsch
  - name: Jan Grashorn
    platform: github
    link: https://github.com/jgrashorn
---
```

# TransportMaps.jl

```@meta
CurrentModule = TransportMaps
```

```@raw html
<p align="center">
  <img src="./assets/logo.svg" alt="TransportMaps.jl logo" width="220"/>
</p>
```

This is an implementation of triangular transport maps in Julia based on the description in [marzouk2016](@cite).
For a comprehensive introduction to transport maps, see [ramgraber2025](@cite). The theoretical foundations for monotone triangular transport maps are detailed in [marzouk2016](@cite), [baptista2023](@cite). For practical applications in structural health monitoring and Bayesian inference, see [grashorn2024](@cite).

## Authors

```@raw html

<Authors />
```

## What are Transport Maps?

Transport maps are smooth, invertible functions that can transform one probability distribution into another [marzouk2016](@cite).
The mathematical foundation builds on the Rosenblatt transformation [rosenblatt1952](@cite) and the Knothe-Rosenblatt rearrangement [knothe1957](@cite).
They are particularly useful for:

- **Sampling**: Generate samples from complex distributions by transforming samples from simple distributions
- **Variational inference**: Approximate complex posterior distributions in Bayesian updating problems [grashorn2024](@cite)
- **Density estimation**: Learn the structure of complex probability distributions

## Key Features

- **Polynomial Maps**: Triangular polynomial transport maps
- **Adaptive Construction**: Automatic selection of polynomial terms for efficient approximation
- **Multiple Rectifiers**: Support for different activation functions (Softplus, ShiftedELU, Identity)
- **Quadrature Integration**: Multiple quadrature schemes for map optimization
- **Optimization**: Built-in optimization routines for fitting maps to target densities
- **Multithreaded evaluation** for processing multiple points efficiently
- **Automatic Differentiation** of target densities with [`DifferentiationInterface.jl`](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/)

## Installation

```julia
using Pkg
Pkg.add("TransportMaps")
```

## Quick Start Example

For a comprehensive introduction, see the page [Getting Started with TransportMaps.jl](@ref), which demonstrates:
- Creating polynomial transport maps
- Setting up quadrature schemes
- Optimizing map parameters
- Generating samples from the learned map

Additional examples are available in the Examples section:
```@contents
Pages = ["Examples/banana_mapfromdensity.md", "Examples/banana_mapfromsamples.md", "Examples/banana_adaptive.md", "Examples/cubic_adaptive_fromdensity.md", "Examples/bod_bayesianinference.md"]
Depth = 1
```

Further, the following manuals discuss the technical details of the implementation:
```@contents
Pages = ["Manuals/basis_functions.md", "Manuals/map_parameterization.md", "Manuals/quadrature_methods.md", "Manuals/optimization.md", "Manuals/conditional_densities.md", "Manuals/adaptive_transport_map.md"]
Depth = 1
```

## Package Architecture

The package is organized around several key components:

### Map Components

- **`PolynomialMapComponent`**: Individual polynomial components of triangular maps
- with different polynomial bases:
  - **`HermiteBasis`**: Probabilists' Hermite polynomial basis
  - **`LinearizedHermiteBasis`**: Linearized Hermite basis for improved conditioning
  - **`CubicSplineHermiteBasis`**: Cubic spline basis with Hermite support
  - **`GaussianWeightedHermiteBasis`**: Hermite basis with Gaussian weighting

### Transport Maps

- **`PolynomialMap`**: Main triangular polynomial transport map
- **`DiagonalMap`**: Convenience constructor for diagonal maps (no cross-terms)
- **`NoMixedMap`**: Maps without mixed polynomial terms
- **`LinearMap`**: Linear standardization map (mean/std)
- **`LaplaceMap`**: Laplace approximation-based map
- **`ComposedMap`**: Composition of linear and polynomial maps

### Rectifier Functions

- **`IdentityRectifier`**: No transformation (linear)
- **`Softplus`**: Smooth positive transformation (log-sum-exp)
- **`ShiftedELU`**: Exponential linear unit variant
- **`ExpRectifier`**: Pure exponential transformation

### Quadrature Schemes

- **`GaussHermiteWeights`**: Gauss-Hermite quadrature points and weights
- **`MonteCarloWeights`**: Monte Carlo integration
- **`LatinHypercubeWeights`**: Latin hypercube sampling
- **`SparseSmolyakWeights`**: Sparse grid quadrature for higher dimensions

### Map Optimization 

- **`optimize!`**: Optimize map coefficients to minimize KL divergence
- **`optimize_adaptive_transportmap`**: Adaptive refinement for map from samples
- **`optimize_adaptive_transportmapcomponent`**: Adaptive refinement for map from density

### Conditional Densities

- **`conditional_density`**: Compute conditional densities π(xₖ | x₁, ..., xₖ₋₁)
- **`conditional_sample`**: Sample from conditional distributions
- **`multivariate_conditional_density`**: Multivariate conditional densities
- **`multivariate_conditional_sample`**: Multivariate conditional sampling

## API Reference

```@contents
Pages = ["api/bases.md", "api/rectifiers.md", "api/densities.md", "api/quadrature.md", "api/maps.md", "api/optimization.md"]
Depth = 1
```

## Related Implementation

- [ATM](https://github.com/baptistar/ATM): Matlab code for adaptive transport maps [baptista2023](@cite)
- [MParT](https://github.com/MeasureTransport/MParT): C++-based library for transport maps [parno2022](@cite)
- [TransportBasedInference.jl](https://github.com/mleprovost/TransportBasedInference.jl): Julia implementation of adaptive transport maps (ATM) and Kalman filters
- [SequentialMeasureTransport.jl](https://github.com/benjione/SequentialMeasureTransport.jl): Julia implementation of transport maps from sum-of-squares densities [zanger2024](@cite)
- [Triangular-Transport-Toolbox](https://github.com/MaxRamgraber/Triangular-Transport-Toolbox): Python code for the triangular transport tutorial paper [ramgraber2025](@cite)
