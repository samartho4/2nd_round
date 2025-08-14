#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

@testset "Property: summarize_ensemble monotonic std" begin
	samples = [[randn(), randn()] for _ in 1:100]
	μ, σ = summarize_ensemble(samples)
	@test all(σ .>= 0)
end 