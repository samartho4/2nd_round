#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Random, Statistics, CSV, DataFrames, Dates, Printf

include(joinpath(@__DIR__, "..", "src", "baseline_models.jl"))
include(joinpath(@__DIR__, "..", "src", "statistical_framework.jl"))

using .BaselineModels
using .StatisticalFramework

println("="^70)
println("COMPREHENSIVE BASELINES – SETUP")
println("="^70)

# This script defines a standardized interface to run and compare multiple baselines
# across identical train/val/test splits. Integrate external toolkits (PySINDy, PySR, PINNs)
# via separate adapters to keep this Julia entrypoint reproducible.

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

struct BaselineResult
	name::String
	mse::Float64
	mae::Float64
	r2::Float64
	train_time_s::Float64
	inference_time_ms::Float64
	params::Int
	memory_mb::Float64
end

function timed(f)
	local t0 = time()
	local x = f()
	return (x, time() - t0)
end

function load_splits()
	train = CSV.read(joinpath(@__DIR__, "..", "data", "train_temporal.csv"), DataFrame)
	val = CSV.read(joinpath(@__DIR__, "..", "data", "val_temporal.csv"), DataFrame)
	test = CSV.read(joinpath(@__DIR__, "..", "data", "test_temporal.csv"), DataFrame)
	return train, val, test
end

function eval_regression_predictions(ŷ::AbstractVector, y::AbstractVector)
	mse = mean((ŷ .- y).^2)
	mae = mean(abs.(ŷ .- y))
	r2 = 1 - sum((y .- ŷ).^2) / sum((y .- mean(y)).^2)
	return mse, mae, r2
end

function run_linear_regression(train::DataFrame, test::DataFrame)
	(model, tt) = timed(() -> train_baseline(LinearRegressionBaseline, train))
	(y1, y2) = begin
		pred = predict(model, test)
		(pred[:,1], pred[:,2])
	end
	m1 = eval_regression_predictions(y1, Vector(test[:, :x1_next]))
	m2 = eval_regression_predictions(y2, Vector(test[:, :x2_next]))
	mse = mean([m1[1], m2[1]])
	mae = mean([m1[2], m2[2]])
	r2 = mean([m1[3], m2[3]])
	return BaselineResult("LinearRegression", mse, mae, r2, tt, 0.0, length(model.β1)+length(model.β2), 0.0)
end

function run_random_forest_stub(train::DataFrame, test::DataFrame)
	(model, tt) = timed(() -> train_baseline(BaselineModels._RFStub, train; n_trees=100))
	pred = predict(model, test)
	m1 = eval_regression_predictions(pred[:,1], Vector(test[:, :x1_next]))
	m2 = eval_regression_predictions(pred[:,2], Vector(test[:, :x2_next]))
	mse = mean([m1[1], m2[1]])
	mae = mean([m1[2], m2[2]])
	r2 = mean([m1[3], m2[3]])
	return BaselineResult("BaggedLinearStub", mse, mae, r2, tt, 0.0, length(model.betas1)+length(model.betas2), 0.0)
end

# Placeholders for external baselines; implement adapters returning BaselineResult
function run_sindy(train::DataFrame, test::DataFrame)
	println("[TODO] Implement SINDy via PyCall/PySINDy adapter.")
	return BaselineResult("SINDy", NaN, NaN, NaN, NaN, NaN, 0, NaN)
end

function run_pysindy_variants(train::DataFrame, test::DataFrame)
	println("[TODO] Implement PySINDy variants STLSQ/SR3.")
	return [BaselineResult("PySINDy-STLSQ", NaN, NaN, NaN, NaN, NaN, 0, NaN),
			BaselineResult("PySINDy-SR3", NaN, NaN, NaN, NaN, NaN, 0, NaN)]
end

function run_pinn(train::DataFrame, test::DataFrame)
	println("[TODO] Implement PINN baseline with the same input/output setup.")
	return BaselineResult("PINN", NaN, NaN, NaN, NaN, NaN, 0, NaN)
end

function run_standard_neural_ode(train::DataFrame, test::DataFrame)
	println("[TODO] Implement standard Neural ODE without UDE residual.")
	return BaselineResult("NeuralODE", NaN, NaN, NaN, NaN, NaN, 0, NaN)
end

function run_sparse_regression(train::DataFrame, test::DataFrame)
	println("[TODO] Implement LASSO/ElasticNet baselines.")
	return [BaselineResult("LASSO", NaN, NaN, NaN, NaN, NaN, 0, NaN),
			BaselineResult("ElasticNet", NaN, NaN, NaN, NaN, NaN, 0, NaN)]
end

function run_gp_physics_prior(train::DataFrame, test::DataFrame)
	println("[TODO] Implement GP baseline with physics priors.")
	return BaselineResult("GP-PhysicsPrior", NaN, NaN, NaN, NaN, NaN, 0, NaN)
end

function run_symbolic_regression(train::DataFrame, test::DataFrame)
	println("[TODO] Implement gplearn/PySR adapters.")
	return [BaselineResult("PySR", NaN, NaN, NaN, NaN, NaN, 0, NaN),
			BaselineResult("DeepSymbolic", NaN, NaN, NaN, NaN, NaN, 0, NaN)]
end

function main()
	train, val, test = load_splits()
	results = BaselineResult[]
	push!(results, run_linear_regression(train, test))
	push!(results, run_random_forest_stub(train, test))
	append!(results, run_pysindy_variants(train, test))
	push!(results, run_sindy(train, test))
	push!(results, run_pinn(train, test))
	push!(results, run_standard_neural_ode(train, test))
	append!(results, run_sparse_regression(train, test))
	push!(results, run_gp_physics_prior(train, test))
	append!(results, run_symbolic_regression(train, test))

	df = DataFrame(
		name = getfield.(results, :name),
		mse = getfield.(results, :mse),
		mae = getfield.(results, :mae),
		r2 = getfield.(results, :r2),
		train_time_s = getfield.(results, :train_time_s),
		inference_time_ms = getfield.(results, :inference_time_ms),
		params = getfield.(results, :params),
		memory_mb = getfield.(results, :memory_mb)
	)
	CSV.write(joinpath(RESULTS_DIR, "comprehensive_baselines.csv"), df)
	println("✅ Saved results to results/comprehensive_baselines.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end 