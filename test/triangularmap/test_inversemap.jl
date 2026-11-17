@testsnippet InverseMapSetup begin
    using TransportMaps
    using Distributions
end

@testitem "InverseMap" setup = [InverseMapSetup] begin

    M = PolynomialMap(2, 2)
    quadrature = GaussHermiteWeights(3, 2)

    banana_density(x) = logpdf(Normal(), (x[1])) + logpdf(Normal(), x[2] - (x[1])^2)
    target = MapTargetDensity(banana_density)

    L = TransportMaps.LaplaceMap(target, [0.0, 0.0])
    newtarget = MapTargetDensity(x -> banana_density(inverse(L, x)))

    optimize!(M, newtarget, quadrature)

    C = L^-1 ∘ M
    Cinv = C^-1
    Cinv2 = L ∘ M^-1
    CCinv = Cinv^-1

    CtimesCinv = C ∘ Cinv

    newsamples = permutedims(rand(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), 1000))

    @test_throws(ArgumentError, TransportMapDistribution(Cinv))
    @test CCinv === C

    @test 1 == 1
end
