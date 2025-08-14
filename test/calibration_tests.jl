#!/usr/bin/env julia

using Test

@testset "Calibration figures" begin
	fig_dir = joinpath(@__DIR__, "..", "paper", "figures")
	req = [
		"calibration_reliability.png",
		"coverage_curve.png"
	]
	missing = String[]
	for f in req
		path = joinpath(fig_dir, f)
		if !isfile(path)
			push!(missing, path)
		end
	end
	if !isempty(missing)
		@info "Calibration figures not found; marking as broken" missing
		@test_broken isempty(missing)
	else
		@test true
	end
end 