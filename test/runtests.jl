using TransportMaps
using Test

import DifferentiationInterface: AutoFiniteDiff, AutoForwardDiff, AutoMooncake, GradientPrep

@testset "TransportMaps.jl" begin
    # Test Map Components
    @testset "Map Components" begin
        include("mapcomponents/univariatebases/test_hermitebasis.jl")
        include("mapcomponents/univariatebases/test_gaussianweighted.jl")
        include("mapcomponents/univariatebases/test_cubicspline.jl")
        include("mapcomponents/univariatebases/test_linearized.jl")
        include("mapcomponents/test_multivariateindices.jl")
        include("mapcomponents/test_multivariatebasis.jl")
        include("mapcomponents/test_rectifier.jl")
        include("mapcomponents/test_polynomialmapcomponent.jl")
    end

    # Test Triangular Maps
    @testset "Triangular Maps" begin
        include("triangularmap/test_polynomialmap.jl")
        include("triangularmap/test_gradients.jl")
        include("triangularmap/test_multithreading.jl")
        include("triangularmap/test_conditionaldensities.jl")
        include("triangularmap/test_linearmap.jl")
        include("triangularmap/test_laplacemap.jl")
        include("triangularmap/test_composedmap.jl")
    end

    # Optimization-related tests (split by source responsibility)
    @testset "Optimization" begin
        include("optimization/test_precomputed_basis.jl")
        include("optimization/test_mapfromdensity.jl")
        include("optimization/test_mapfromsamples.jl")
        include("optimization/test_optimizationhistory.jl")
        include("optimization/test_adaptivetransportmap.jl")
        include("optimization/test_adaptivetransportmap_fromdensity.jl")
    end

    @testset "Quadrature" begin
        include("quadrature/test_quadratureknots.jl")
        include("quadrature/test_quadratureweights.jl")
    end

    # Test Utilities
    @testset "Utilities" begin
        include("util/test_gaussquadrature.jl")
        include("util/test_hybridrootfinder.jl")
        include("util/test_mapdensity.jl")
    end
end
