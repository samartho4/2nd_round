#!/usr/bin/env julia

using Test

@testset "Calibration figures" begin
	ok = true
	try
		include(joinpath(@__DIR__, "..", "scripts", "generate_figures.jl"))
	catch e
		ok = false
	end
	@test ok
	@test isfile(joinpath(@__DIR__, "..", "paper", "figures", "calibration_reliability.png"))
	@test isfile(joinpath(@__DIR__, "..", "paper", "figures", "coverage_curve.png"))
end 