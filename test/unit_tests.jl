#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

@testset "Neural ODE architectures" begin
	dx = zeros(2)
	x = [0.1, -0.2]
	p = ones(10)
	NeuralNODEArchitectures.baseline_nn!(dx, x, p, 0.0)
	@test length(dx) == 2
end 