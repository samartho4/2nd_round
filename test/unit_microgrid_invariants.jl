#!/usr/bin/env julia

using Test, SHA, CSV, DataFrames
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
using .Microgrid

@testset "Microgrid invariants" begin
	dx = zeros(2)
	x = [0.5, 0.0]
	p = (0.9, 0.9, 0.3, 1.2, 0.4)
	Microgrid.microgrid!(dx, x, p, 0.0)
	@test isfinite(dx[1]) && isfinite(dx[2])
	# Units/sign checks: when no generation and nominal load, Pnet <= 0
	# so β*Pnet term should be <= 0 and dx2 should decrease absent γ*x1
	Pnet0 = Microgrid.generation(0.0) - Microgrid.load(0.0)
	@test Pnet0 <= 0
	Microgrid.microgrid!(dx, x, p, 0.0)
	@test dx[2] <= p[4]*Pnet0 + p[5]*x[1]  # -α*x2 term can't increase above phys forcing
end

@testset "Deterministic dataset hashes" begin
	# Verify current files match recorded hashes if present
	hashfile = joinpath(@__DIR__, "..", "data", "hashes.txt")
	if isfile(hashfile)
		lines = readlines(hashfile)
		mismatch = String[]
		for ln in lines
			parts = split(strip(ln))
			length(parts) == 2 || continue
			h, rel = parts
			path = joinpath(@__DIR__, "..", "data", rel)
			if isfile(path)
				buf = read(path)
				calc = bytes2hex(sha1(buf))
				if lowercase(calc) != lowercase(h)
					push!(mismatch, rel)
				end
			end
		end
		@test isempty(mismatch)
	else
		@info "No hashes.txt present; skipping checksum test"
		@test true
	end
end 