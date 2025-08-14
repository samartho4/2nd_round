#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Random, Statistics, CSV, DataFrames, LinearAlgebra
using Printf, Dates

include(joinpath(@__DIR__, "..", "src", "physics_discovery_validation.jl"))

println("="^70)
println("PHYSICS MEANINGFULNESS VERIFICATION")
println("="^70)

function dimensional_analysis()
	println("[TODO] Add dimensional analysis based on variable units.")
	return true
end

function energy_conservation_check()
	println("[TODO] Compute energy proxy and check conservation/stability.")
	return Dict("holds"=>false)
end

function lyapunov_stability_check()
	println("[TODO] Approximate Lyapunov function / local stability via Jacobian eigs.")
	return Dict("stable"=>false)
end

function sensitivity_to_initial_conditions()
	println("[TODO] Perturb initial conditions and measure divergence.")
	return Dict("sensitivity"=>NaN)
end

function main()
	model_path = joinpath(@__DIR__, "..", "checkpoints", "ude_model.jld2")
	if !isfile(model_path)
		println("Model not found at $(model_path)")
		return
	end
	ude_model = load_trained_model(model_path)
	scenarios = Glob.glob(joinpath(@__DIR__, "..", "data", "scenarios", "*/test.csv"))
	core = validate_physics_discovery(ude_model, scenarios)
	add = Dict(
		"dimensional_analysis" => dimensional_analysis(),
		"energy_conservation" => energy_conservation_check(),
		"lyapunov" => lyapunov_stability_check(),
		"sensitivity" => sensitivity_to_initial_conditions()
	)
	JLD2.save(joinpath(@__DIR__, "..", "results", "physics_validation_full.jld2"), "core", core, "extra", add)
	println("✅ Saved results/physics_validation_full.jld2")
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end 