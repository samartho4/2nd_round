#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
using .Microgrid

@testset "Numerical: microgrid local derivative" begin
	dx = zeros(2)
	x = [0.1, -0.2]
	p = (0.9, 0.9, 0.3, 1.2, 0.4)
	Microgrid.microgrid!(dx, x, p, 0.0)
	@test isfinite(dx[1]) && isfinite(dx[2])
end 