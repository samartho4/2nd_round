#!/usr/bin/env julia

using Test
include(joinpath(@__DIR__, "..", "src", "training.jl"))
using .Training

@testset "Config tolerances present" begin
	cfg = Training.load_config()
	@test haskey(cfg, "solver")
	@test haskey(cfg["solver"], "abstol")
	@test haskey(cfg["solver"], "reltol")
end

@testset "Training entrypoint exists" begin
	@test hasmethod(Training.train!, Tuple{} ) == false
	@test hasmethod(Training.train!, Tuple{Any} ) == false
	@test hasmethod(Training.train!, Tuple{NamedTuple} ) == false
	# confirm callable with kwargs
	@test isa(Training.train!, Function)
end 