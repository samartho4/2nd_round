#!/usr/bin/env julia

using Test
using CSV, DataFrames

@testset "Baselines adapters" begin
	ok = true
	try
		include(joinpath(@__DIR__, "..", "scripts", "comprehensive_baselines.jl"))
	catch e
		ok = false
	end
	@test ok
	@test isfile(joinpath(@__DIR__, "..", "results", "comprehensive_baselines.csv"))
	df = CSV.read(joinpath(@__DIR__, "..", "results", "comprehensive_baselines.csv"), DataFrame)
	@test nrow(df) >= 5
end 