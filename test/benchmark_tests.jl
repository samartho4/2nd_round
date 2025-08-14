#!/usr/bin/env julia

using Test
using BenchmarkTools
 
@testset "Benchmark: trivial" begin
	bt = @belapsed sum(abs, rand(1000))
	@test bt > 0
end 