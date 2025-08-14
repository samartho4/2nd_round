#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Random, Statistics, CSV, DataFrames, Dates, Printf

println("="^70)
println("COMPREHENSIVE ABLATION STUDIES – SCAFFOLD")
println("="^70)

# Define grids
HIDDEN_DIMS = [2,4,8,16,32]
DEPTHS = [1,2,3,4]
ACTS = ["tanh","relu","elu","swish","sin"]
OPTIMS = ["Adam","AdamW","RMSprop","BFGS"]
LRS = [1e-3, 5e-4, 1e-4]
BATCHES = [32, 64, 128, 256]
EPOCHS = [50, 100, 200]
POLY_DEG = [1,2,3,4,5]

function run_config(cfg::Dict)
	# TODO: integrate training call for the specific configuration
	# Return a NamedTuple with metrics
	return (mse = NaN, train_time_s = NaN, params = NaN)
end

function main()
	mkpath(joinpath(@__DIR__, "..", "results"))
	rows = DataFrame()
	for h in HIDDEN_DIMS, d in DEPTHS, a in ACTS, opt in OPTIMS, lr in LRS, b in BATCHES, e in EPOCHS
		cfg = Dict(
			"hidden"=>h, "depth"=>d, "act"=>a, "optim"=>opt, "lr"=>lr, "batch"=>b, "epochs"=>e
		)
		res = run_config(cfg)
		push!(rows, (
			hidden=h, depth=d, act=a, optim=opt, lr=lr, batch=b, epochs=e,
			mse=res.mse, train_time_s=res.train_time_s, params=res.params
		))
	end
	CSV.write(joinpath(@__DIR__, "..", "results", "ablation_arch_training.csv"), rows)
	println("✅ Saved results/ablation_arch_training.csv")

	rows2 = DataFrame()
	for deg in POLY_DEG
		cfg = Dict("poly_degree"=>deg)
		res = run_config(cfg)
		push!(rows2, (poly_degree=deg, mse=res.mse))
	end
	CSV.write(joinpath(@__DIR__, "..", "results", "ablation_physics_polydeg.csv"), rows2)
	println("✅ Saved results/ablation_physics_polydeg.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end 