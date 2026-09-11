# TransportMaps.jl

[![Build Status](https://github.com/JuliaUQ/TransportMaps.jl/actions/workflows/juliaci.yml/badge.svg?branch=main)](https://github.com/JuliaUQ/TransportMaps.jl/actions/workflows/juliaci.yml?query=branch:main)
[![codecov](https://codecov.io/github/JuliaUQ/TransportMaps.jl/graph/badge.svg?token=PQTR0PG87A)](https://codecov.io/github/JuliaUQ/TransportMaps.jl)
[![doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliauq.github.io/TransportMaps.jl/dev/)
[![doc-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliauq.github.io/TransportMaps.jl/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/885372338.svg)](https://doi.org/10.5281/zenodo.18309940)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

A Julia implementation of triangular transport maps for variational inference.

## Quick Start

### Installation

```julia
# From Julia REPL
pkg> add TransportMaps
```

### Getting Started

Here's a simple example showing how to construct a transport map for a "banana" distribution:

```julia
using TransportMaps
using Distributions

# Create a 2D polynomial map with degree 2 and Softplus rectifier
M = PolynomialMap(2, 2, Normal(), Softplus())

# Set up quadrature for optimization
quadrature = GaussHermiteWeights(3, 2)

# Define target density (banana distribution)
target_density(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
target = MapTargetDensity(target_density)

# Optimize the map coefficients
result = optimize!(M, target, quadrature)

# Generate samples by mapping standard Gaussian samples
samples_z = randn(1000, 2)

# Matrix input automatically uses multithreading for better performance
mapped_samples = evaluate(M, samples_z)

# Evaluate map quality
variance_diag = variance_diagnostic(M, target, samples_z)
```

## Features

- **Triangular polynomial transport maps** with various polynomial bases
- **Multiple rectifier functions**: Softplus, ShiftedELU, Identity
- **Quadrature integration schemes**: Gauss-Hermite, Monte Carlo, Latin Hypercube
- **Automatic optimization** of map coefficients via KL divergence minimization
- **Multithreaded evaluation** for processing multiple points efficiently
- **Matrix input support** for all core functions (evaluate, inverse, jacobian, etc.)

Please refer to the [documentation](https://juliauq.github.io/TransportMaps.jl/dev/) for more extensive examples and explanations.

## Related Implementation

Related implementations of transport maps are:

- [ATM](https://github.com/baptistar/ATM): Matlab code for adaptive transport maps [1]
- [MParT](https://github.com/MeasureTransport/MParT): C++-based library for transport maps [2]
- [TransportBasedInference.jl](https://github.com/mleprovost/TransportBasedInference.jl): Julia implementation of adaptive transport maps (ATM) and Kalman filters
- [SequentialMeasureTransport.jl](https://github.com/benjione/SequentialMeasureTransport.jl): Julia implementation of transport maps from sum-of-squares densities [3]
- [Triangular-Transport-Toolbox](https://github.com/MaxRamgraber/Triangular-Transport-Toolbox): Python code for the triangular transport tutorial paper [4]

### References

1. Baptista, R., Marzouk, Y., & Zahm, O. (2023). On the Representation and Learning of Monotone Triangular Transport Maps. Foundations of Computational Mathematics. https://doi.org/10.1007/s10208-023-09630-x
2. Parno, M., Rubio, P.-B., Sharp, D., Brennan, M., Baptista, R., Bonart, H., & Marzouk, Y. (2022). MParT: Monotone Parameterization Toolkit. Journal of Open Source Software, 7(80), 4843. https://doi.org/10.21105/joss.04843
3. Zanger, B., Zahm, O., Cui, T., & Schreiber, M. (2024). Sequential transport maps using SoS density estimation and $α$-divergences (No. arXiv:2402.17943). arXiv. https://doi.org/10.48550/arXiv.2402.17943
4. Ramgraber, M., Sharp, D., Provost, M. L., & Marzouk, Y. (2025). A friendly introduction to triangular transport (No. arXiv:2503.21673). arXiv. https://doi.org/10.48550/arXiv.2503.21673

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TransportMaps.jl in your research, please cite using the information provided at `Cite this repository` (or see [`CITATION.cff`](CITATION.cff)).
