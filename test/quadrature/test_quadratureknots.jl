@testsnippet QuadratureKnotsSetup begin
    using TransportMaps
    using Test
    using Distributions
    using FastGaussQuadrature

end

@testitem "Quadrature Knots" setup = [QuadratureKnotsSetup] begin
    gh = GaussHermiteKnots()
    @test support(gh) == RealInterval(-Inf, Inf)

    p_0, w_0 = gh(0)
    @test p_0 == [0.0]
    @test w_0 == [1.0]

    for l in 1:4
        p_l, w_l = gh(l)
        n = 2^l + 1
        p_test, w_test = gausshermite(n; normalize = true)
        @test p_l == p_test
        @test w_l == w_test
        @test length(p_test) == n
    end

    gh_nonstandard = GaussHermiteKnots(Normal(2, 3))
    p_nonstandard, w_nonstandard = gh_nonstandard(3)
    @test sum(w_nonstandard .* p_nonstandard) ≈ 2.0
    @test sum(w_nonstandard .* (p_nonstandard .- 2.0) .^ 2) ≈ 9.0

    p_0, w_0 = gh_nonstandard(0)
    @test p_0 == [2.0]
    @test w_0 == [1.0]

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
