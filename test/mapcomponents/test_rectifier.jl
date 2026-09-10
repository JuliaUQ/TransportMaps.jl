@testsnippet RectifierSetup begin
    using TransportMaps
    using Test

end

@testitem "Rectifier Functions" setup = [RectifierSetup] begin
    @testset "Softplus" begin
        softplus = Softplus()

        # Test basic functionality
        @test softplus(0.0) ≈ log(2.0)  # log(1 + exp(0)) = log(2)
        @test softplus(-10.0) ≈ exp(-10.0) atol = 1.0e-7  # log(1 + exp(-10)) ≈ exp(-10) for large negative values
        @test softplus(10.0) ≈ 10.0 atol = 1.0e-4  # log(1 + exp(10)) ≈ 10 for large positive values

        # Test monotonicity (should be increasing)
        @test softplus(-1.0) < softplus(0.0) < softplus(1.0)

        # Test positivity
        @test softplus(-100.0) > 0.0
        @test softplus(0.0) > 0.0
        @test softplus(100.0) > 0.0

        # Test type stability
        @test softplus(1) isa Float64
        @test softplus(1.0) isa Float64

        # Test softplus with different β values
        softplus_beta_2 = Softplus(2.0)
        @test softplus_beta_2(0.0) ≈ log(2.0) / 2.0
        @test softplus_beta_2(-10.0) ≈ exp(-20.0) / 2.0 atol = 1.0e-9
        @test softplus_beta_2(10.0) ≈ 10.0 atol = 1.0e-4

        for x in (-2.0, 0.0, 2.0)
            finite_difference = TransportMaps.FiniteDiff.finite_difference_derivative(
                y -> TransportMaps.derivative(softplus_beta_2, y), x
            )
            @test TransportMaps.second_derivative(softplus_beta_2, x) ≈
                finite_difference rtol = 1.0e-6
        end

    end

    @testset "ShiftedELU" begin
        shifted_elu = ShiftedELU()

        # Test basic functionality
        @test shifted_elu(0.0) ≈ 1.0  # exp(0) = 1 for ξ ≤ 0
        @test shifted_elu(-1.0) ≈ exp(-1.0)  # exp(-1) for ξ ≤ 0
        @test shifted_elu(1.0) ≈ 2.0  # 1 + 1 = 2 for ξ > 0
        @test shifted_elu(2.0) ≈ 3.0  # 2 + 1 = 3 for ξ > 0

        # Test continuity at ξ = 0
        @test shifted_elu(0.0) ≈ 1.0
        @test abs(shifted_elu(1.0e-10) - shifted_elu(-1.0e-10)) < 1.0e-9

        # Test monotonicity (should be increasing)
        @test shifted_elu(-2.0) < shifted_elu(-1.0) < shifted_elu(0.0) < shifted_elu(1.0) < shifted_elu(2.0)

        # Test positivity
        @test shifted_elu(-100.0) > 0.0
        @test shifted_elu(0.0) > 0.0
        @test shifted_elu(100.0) > 0.0

        @test TransportMaps.second_derivative(shifted_elu, -1.0) ≈ exp(-1.0)
        @test TransportMaps.second_derivative(shifted_elu, 0.0) == 0.5
        @test TransportMaps.second_derivative(shifted_elu, 1.0) == 0.0

        # Test type stability
        @test shifted_elu(1) isa Real
        @test shifted_elu(1.0) isa Float64
    end

    @testset "IdentityRectifier" begin
        identity_rect = IdentityRectifier()

        # Test basic functionality
        @test identity_rect(0.0) ≈ 0.0
        @test identity_rect(1.0) ≈ 1.0
        @test identity_rect(-1.0) ≈ -1.0
        @test identity_rect(π) ≈ π
        @test TransportMaps.second_derivative(identity_rect, 1.0) == 0.0

        # Test linearity
        @test identity_rect(2.0 * 3.0) ≈ 2.0 * identity_rect(3.0)
        @test identity_rect(1.0 + 2.0) ≈ identity_rect(1.0) + identity_rect(2.0)

        # Test type stability
        @test identity_rect(1) isa Real
        @test identity_rect(1.0) isa Float64
    end

    @testset "ExpRectifier" begin

        exponential_rect = TransportMaps.ExpRectifier()

        # Test basic functionality
        @test exponential_rect(0.0) ≈ 1.0
        @test exponential_rect(2.0) ≈ exp(2.0)
        @test exponential_rect(-3.0) ≈ exp(-3.0)

        # Test monotonicity
        @test exponential_rect(-1.0) < exponential_rect(0.0) < exponential_rect(1.0) < exponential_rect(2.0)

        # Test derivative
        @test TransportMaps.derivative(exponential_rect, 0.0) ≈ exp(0.0)
        @test TransportMaps.derivative(exponential_rect, 1.0) ≈ exp(1.0)
        @test TransportMaps.derivative(exponential_rect, -1.0) ≈ exp(-1.0)
        @test TransportMaps.second_derivative(exponential_rect, 1.0) ≈ exp(1.0)

        # Test type stability
        @test exponential_rect(1) isa Real
        @test exponential_rect(1.0) isa Float64
    end

    @testset "Show" begin
        sp = Softplus()
        se = ShiftedELU()
        ir = IdentityRectifier()

        @test_nowarn sprint(show, sp)
        @test_nowarn sprint(print, sp)
        @test_nowarn display(sp)

        @test_nowarn sprint(show, se)
        @test_nowarn sprint(print, se)
        @test_nowarn display(se)

        @test_nowarn sprint(show, ir)
        @test_nowarn sprint(print, ir)
        @test_nowarn display(ir)
    end
end
