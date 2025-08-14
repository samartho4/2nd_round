#!/usr/bin/env julia

using Test
 
@testset "Integration: data and figures" begin
	@test isfile(joinpath(@__DIR__, "..", "data", "test_dataset.csv"))
end 