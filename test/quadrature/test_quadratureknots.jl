using TransportMaps
using Test
using Distributions
using FastGaussQuadrature

@testset "Quadrature Knots" begin
    gh = GaussHermiteKnots()
    @test support(gh) == RealInterval(-Inf, Inf)

    p_0, w_0 = gh(0)
    @test p_0 == [0.0]
    @test w_0 == [1.0]

    for l in 1:4
        p_l, w_l = gh(l)
        n = 2^l + 1
        p_test, w_test = gausshermite(n; normalize=true)
        @test p_l == p_test
        @test w_l == w_test
        @test length(p_test) == n
    end

    gl = GaussLegendreKnots()
    @test support(gl) == RealInterval(-1, 1)

    p_0, w_0 = gl(0)
    @test p_0 == [0.0]
    @test w_0 == [1.0]

    for l in 1:4
        p_l, w_l = gl(l)
        n = 2^l + 1
        p_test, w_test = gausslegendre(n)
        @test p_l == p_test
        @test w_l == w_test ./ 2
        @test length(p_test) == n
    end

    cc = ClenshawCurtisKnots()
    @test support(cc) == RealInterval(-1, 1)

    p_0, w_0 = cc(0)
    @test p_0 == [0.0]
    @test w_0 == [1.0]

    for l in 1:4
        p_l, w_l = cc(l)
        p_test, w_test = TransportMaps.clenshaw_curtis_rule(2^l)
        @test p_l == p_test
        @test w_l == w_test ./ 2
        @test length(p_test) == 2^l + 1
    end

    # Test transformation to [0, 1]
    gl01 = GaussLegendreKnots([0, 1])
    @test support(gl01) == RealInterval(0, 1)

    p_0, w_0 = gl01(0)
    @test p_0 == [0.5]
    @test w_0 == [1.0]

    for l in 1:4
        p_l, w_l = gl01(l)
        n = 2^l + 1
        p_test, w_test = gausslegendre(n)
        p_test .= 0.5 .* p_test .+ 0.5
        w_test .= w_test ./ 2
        @test p_l == p_test
        @test w_l == w_test
        @test length(p_test) == n
    end

    cc01 = ClenshawCurtisKnots([0, 1])
    @test support(cc01) == RealInterval(0, 1)

    p_0, w_0 = cc01(0)
    @test p_0 == [0.5]
    @test w_0 == [1.0]

    for l in 1:4
        p_l, w_l = cc01(l)
        p_test, w_test = TransportMaps.clenshaw_curtis_rule(2^l)
        p_test .= 0.5 .* p_test .+ 0.5
        w_test .= w_test ./ 2
        @test p_l == p_test
        @test w_l == w_test
        @test length(p_test) == 2^l + 1
    end
end
