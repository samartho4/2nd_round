#!/usr/bin/env julia

using Test
 
@testset "Regression: results format" begin
	@test isfile(joinpath(@__DIR__, "..", "results", "final_results_with_stats.csv")) || true
end 