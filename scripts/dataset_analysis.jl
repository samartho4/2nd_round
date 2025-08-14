#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Random, Statistics, CSV, DataFrames, Dates, Printf

include(joinpath(@__DIR__, "..", "src", "statistical_framework.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline_models.jl"))

using .BaselineModels
using .StatisticalFramework

println("="^70)
println("DATASET SCALING & VALIDATION ANALYSIS")
println("="^70)

const SIZES = [100, 250, 500, 1000, 1500, 3000, 5000]

function load_full_train_val_test()
	train = CSV.read(joinpath(@__DIR__, "..", "data", "train_temporal.csv"), DataFrame)
	val = CSV.read(joinpath(@__DIR__, "..", "data", "val_temporal.csv"), DataFrame)
	test = CSV.read(joinpath(@__DIR__, "..", "data", "test_temporal.csv"), DataFrame)
	return train, val, test
end

function subsample_rows(df::DataFrame, n::Int; seed=42)
	Random.seed!(seed)
	idx = collect(1:nrow(df))
	shuffle!(idx)
	n = min(n, nrow(df))
	return df[idx[1:n], :]
end

function evaluate_linear_multi(train::DataFrame, test::DataFrame)
	model = train_baseline(LinearRegressionBaseline, train)
	pred = predict(model, test)
	mse = mean(mean((pred .- Matrix(test[:, [:x1_next, :x2_next]])).^2; dims=2))
	return mse
end

function learning_curves()
	train, val, test = load_full_train_val_test()
	rows = DataFrame(size=Int[], mse=Float64[])
	for n in SIZES
		tr = subsample_rows(train, n)
		mse = evaluate_linear_multi(tr, test)
		push!(rows, (n, mse))
		println(@sprintf("n=%d -> MSE=%.4f", n, mse))
	end
	CSV.write(joinpath(@__DIR__, "..", "results", "learning_curve_linear.csv"), rows)
	println("✅ Saved results/learning_curve_linear.csv")
end

function kfold_cv(df::DataFrame; k::Int=5)
	n = nrow(df)
	idx = collect(1:n)
	folds = [idx[i:k:n] for i in 1:k]
	scores = Float64[]
	for i in 1:k
		test_idx = folds[i]
		train_idx = setdiff(idx, test_idx)
		tr = df[train_idx, :]
		te = df[test_idx, :]
		try
			m = evaluate_linear_multi(tr, te)
			push!(scores, m)
		catch e
			println("CV fold failed: ", e)
		end
	end
	return scores
end

function walk_forward_validation(df::DataFrame; n_splits::Int=5)
	sort!(df, :time)
	n = nrow(df)
	split_points = [round(Int, i * n/(n_splits+1)) for i in 1:n_splits]
	scores = Float64[]
	for sp in split_points
		tr = df[1:sp, :]
		te = df[sp+1:end, :]
		if nrow(te) < 2; continue; end
		try
			m = evaluate_linear_multi(tr, te)
			push!(scores, m)
		catch e
			println("WF split failed: ", e)
		end
	end
	return scores
end

function main()
	train, val, test = load_full_train_val_test()
	mkpath(joinpath(@__DIR__, "..", "results"))
	learning_curves()
	cv_scores = kfold_cv(vcat(train, val); k=5)
	wf_scores = walk_forward_validation(vcat(train, val); n_splits=5)
	open(joinpath(@__DIR__, "..", "results", "dataset_validation_summary.md"), "w") do f
		println(f, "# Dataset Validation Summary")
		println(f, "\n## K-Fold CV (k=5)")
		println(f, @sprintf("Mean MSE=%.4f ± %.4f", mean(cv_scores), std(cv_scores)))
		println(f, "\n## Walk-Forward Validation")
		println(f, @sprintf("Mean MSE=%.4f ± %.4f", mean(wf_scores), std(wf_scores)))
	end
	println("✅ Saved results/dataset_validation_summary.md")
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end 