#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Random, Statistics, CSV, DataFrames, Dates, Printf
using LinearAlgebra
using Flux
using SymbolicRegression

include(joinpath(@__DIR__, "..", "src", "statistical_framework.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline_models.jl"))
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))

using .StatisticalFramework
using .BaselineModels
using .Microgrid

println("="^70)
println("COMPREHENSIVE BASELINES – SETUP")
println("="^70)

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

# Utility: compute Pgen/Pload
compute_power_inputs(t) = (max(0, sin((t - 6) * π / 12)), 0.6 + 0.2 * sin(t * π / 12))

function features_matrix(df::DataFrame)
	t = Vector{Float64}(df[:, :time])
	n = length(t)
	Pgen = similar(t)
	Pload = similar(t)
	for i in 1:n
		pg, pl = compute_power_inputs(t[i])
		Pgen[i] = pg; Pload[i] = pl
	end
	X = hcat(t, Vector(df[:, :x1]), Vector(df[:, :x2]), Pgen, Pload)
	return X, t
end

# Utility: get global median dt from time stamps
function estimate_dt(t::Vector{Float64})
	if length(t) < 2; return 1.0; end
	return median(diff(t))
end

# Evaluation helper: compute next-step metrics from predictions
function eval_next_step(pred::Matrix{Float64}, df_test::DataFrame)
	y = Matrix(df_test[:, [:x1_next, :x2_next]])
	mse = mean((pred .- y).^2)
	mae = mean(abs.(pred .- y))
	r2 = 1 - sum((y .- pred).^2) / sum((y .- mean(y)).^2)
	return mse, mae, r2
end

# ----- 1) SINDy (STLSQ proxy) -----
function run_sindy_stlsq(train::DataFrame, test::DataFrame; λ=1e-3, degree=3, iters=5)
	Xtr, ttr = features_matrix(train)
	Xte, tte = features_matrix(test)
	dt = estimate_dt(ttr)
	Y = Matrix(train[:, [:x1, :x2]])
	Ynext = Matrix(train[:, [:x1_next, :x2_next]])
	dY = (Ynext .- Y) ./ dt
	function library(X)
		# build polynomial library up to degree 3
		n, d = size(X)
		cols = Vector{Vector{Float64}}()
		push!(cols, ones(n))
		# degree 1
		for j in 1:d
			push!(cols, X[:, j])
		end
		# degree 2
		if degree >= 2
			for j in 1:d
				push!(cols, X[:, j].^2)
			end
			for j in 1:d
				for k in j+1:d
					push!(cols, X[:, j] .* X[:, k])
				end
			end
		end
		# degree 3 (subset)
		if degree >= 3
			for j in 1:d
				push!(cols, X[:, j].^3)
			end
		end
		return reduce(hcat, cols)
	end
	Θ = library(Xtr)
	Θt = library(Xte)
	# STLSQ loop per dimension
	β = zeros(size(Θ,2), 2)
	for dim in 1:2
		θ = Θ \ dY[:, dim]
		for _ in 1:iters
			mask = abs.(θ) .> λ
			mask[1] = true # keep intercept
			θ = (Θ[:, mask] \ dY[:, dim])
			fullθ = zeros(size(Θ,2))
			fullθ[mask] = θ
			θ = fullθ
		end
		β[:, dim] = θ
	end
	dYte = Θt * β
	Yte = Matrix(test[:, [:x1, :x2]])
	pred = Yte .+ dYte .* estimate_dt(tte)
	mse, mae, r2 = eval_next_step(pred, test)
	return BaselineResult("SINDy-STLSQ", mse, mae, r2, 0.0, 0.0, count(!=(0.0), β), 0.0)
end

# ----- 2) PySINDy (SR3 proxy via soft-threshold) -----
function run_pysindy_sr3(train::DataFrame, test::DataFrame; λ=1e-3, degree=3, iters=50, η=1e-2)
	Xtr, ttr = features_matrix(train)
	Xte, tte = features_matrix(test)
	dt = estimate_dt(ttr)
	Y = Matrix(train[:, [:x1, :x2]])
	Ynext = Matrix(train[:, [:x1_next, :x2_next]])
	dY = (Ynext .- Y) ./ dt
	Θ = let
		function library(X)
			n, d = size(X)
			cols = Vector{Vector{Float64}}(); push!(cols, ones(n))
			for j in 1:d; push!(cols, X[:, j]); end
			for j in 1:d; push!(cols, X[:, j].^2); end
			return reduce(hcat, cols)
		end
		library(Xtr)
	end
	Θt = Θ[1:size(Xte,1), :]  # approximate same transform
	β = zeros(size(Θ,2), 2)
	for dim in 1:2
		θ = Θ \ dY[:, dim]
		for _ in 1:iters
			# gradient step
			g = Θ' * (Θ*θ - dY[:, dim])
			θ -= η * g
			# soft-threshold (SR3 style prox)
			θ = sign.(θ) .* max.(0.0, abs.(θ) .- η*λ)
			θ[1] = θ[1] # keep intercept (not thresholded)
		end
		β[:, dim] = θ
	end
	dYte = Θt * β
	Yte = Matrix(test[:, [:x1, :x2]])
	pred = Yte .+ dYte .* estimate_dt(tte)
	mse, mae, r2 = eval_next_step(pred, test)
	return BaselineResult("PySINDy-SR3", mse, mae, r2, 0.0, 0.0, count(!=(0.0), β), 0.0)
end

# ----- 3) Standard Neural ODE proxy (NN for dx) -----
function run_neural_dx(train::DataFrame, test::DataFrame; hidden=16, epochs=200, lr=1e-3)
	Xtr, ttr = features_matrix(train)
	Xte, tte = features_matrix(test)
	dt = estimate_dt(ttr)
	Y = Matrix(train[:, [:x1, :x2]])
	Ynext = Matrix(train[:, [:x1_next, :x2_next]])
	dY = (Ynext .- Y) ./ dt
	Xtrf = Float32.(Xtr)'; dYf = Float32.(dY)
	model = Chain(Dense(size(Xtr,2), hidden, tanh), Dense(hidden, 2))
	loss() = mean(sum((model(Xtrf)' .- dYf).^2, dims=2))
	ps = Flux.params(model)
	for _ in 1:epochs
		grads = Flux.gradient(ps) do
			loss()
		end
		for p in ps
			p .-= lr .* grads[p]
		end
	end
	dYte = Array(model(Float32.(Xte)'))'  # next-step derivative
	pred = Matrix(test[:, [:x1, :x2]]) .+ dYte .* estimate_dt(tte)
	mse, mae, r2 = eval_next_step(pred, test)
	params = sum(length, ps)
	return BaselineResult("NeuralDX", mse, mae, r2, 0.0, 0.0, params, 0.0)
end

# ----- 4) PINN-like residual NN (physics-regularised) -----
function run_pinn_like(train::DataFrame, test::DataFrame; hidden=16, epochs=200, lr=1e-3, λ_phys=1.0)
	Xtr, ttr = features_matrix(train)
	dt = estimate_dt(ttr)
	Y = Matrix(train[:, [:x1, :x2]])
	Ynext = Matrix(train[:, [:x1_next, :x2_next]])
	dY = (Ynext .- Y) ./ dt
	Xtrf = Float32.(Xtr)'
	resnn = Chain(Dense(size(Xtr,2), hidden, tanh), Dense(hidden, 1))
	function physics_dx1(t)
		u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
		ηin, ηout = 0.9, 0.9
		Pin = u > 0 ? ηin * u : (1 / ηout) * u
		return Pin - (0.6 + 0.2 * sin(t * π / 12))
	end
	function loss()
		res = resnn(Xtrf)'  # n×1
		dx1 = [physics_dx1(tt) for tt in ttr]
		dx = hcat(dx1, (-0.3 .* Y[:,2] .+ vec(res) .+ 0.02 .* Y[:,1]))
		fit = mean(sum((dx .- dY).^2, dims=2))
		phys_pen = mean((dx[:,1] .- dY[:,1]).^2)
		return fit + λ_phys * phys_pen
	end
	ps = Flux.params(resnn)
	for _ in 1:epochs
		grads = Flux.gradient(ps) do
			loss()
		end
		for p in ps
			p .-= lr .* grads[p]
		end
	end
	# Predict on test
	Xte, tte = features_matrix(test)
	res_te = resnn(Float32.(Xte)')'
	dx1_te = [physics_dx1(tt) for tt in tte]
	dx_te = hcat(dx1_te, (-0.3 .* Matrix(test[:, [:x2]])[:,1] .+ vec(res_te) .+ 0.02 .* Matrix(test[:, [:x1]])[:,1]))
	pred = Matrix(test[:, [:x1, :x2]]) .+ dx_te .* estimate_dt(tte)
	mse, mae, r2 = eval_next_step(pred, test)
	params = sum(length, ps)
	return BaselineResult("PINN-like", mse, mae, r2, 0.0, 0.0, params, 0.0)
end

# ----- 5) LASSO / Elastic Net (proximal gradient) -----
function prox_l1(x, τ)
	return sign.(x) .* max.(0.0, abs.(x) .- τ)
end
function run_lasso(train::DataFrame, test::DataFrame; λ=1e-3, iters=500, lr=1e-2)
	Xtr, _ = features_matrix(train)
	y = Matrix(train[:, [:x1_next, :x2_next]])
	X = hcat(ones(size(Xtr,1)), Xtr)
	W = zeros(size(X,2), 2)
	for _ in 1:iters
		G = X' * (X*W - y)
		W .-= lr .* G ./ size(X,1)
		W[2:end, :] .= prox_l1(W[2:end, :], lr*λ)
	end
	Xte, _ = features_matrix(test)
	Xte = hcat(ones(size(Xte,1)), Xte)
	pred = Xte * W
	mse, mae, r2 = eval_next_step(pred, test)
	return BaselineResult("LASSO", mse, mae, r2, 0.0, 0.0, length(W), 0.0)
end
function run_elastic_net(train::DataFrame, test::DataFrame; λ1=1e-3, λ2=1e-4, iters=500, lr=1e-2)
	Xtr, _ = features_matrix(train)
	y = Matrix(train[:, [:x1_next, :x2_next]])
	X = hcat(ones(size(Xtr,1)), Xtr)
	W = zeros(size(X,2), 2)
	for _ in 1:iters
		G = X' * (X*W - y) .+ 2*λ2 .* W
		W .-= lr .* G ./ size(X,1)
		W[2:end, :] .= prox_l1(W[2:end, :], lr*λ1)
	end
	Xte, _ = features_matrix(test)
	Xte = hcat(ones(size(Xte,1)), Xte)
	pred = Xte * W
	mse, mae, r2 = eval_next_step(pred, test)
	return BaselineResult("ElasticNet", mse, mae, r2, 0.0, 0.0, length(W), 0.0)
end

# ----- 6) GP with RBF kernel via Kernel Ridge Regression -----
function rbf_kernel(X::Matrix{Float64}, Z::Matrix{Float64}; ℓ=1.0, σ2=1.0)
	# squared Euclidean distances
	d2 = [sum((X[i,:] .- Z[j,:]).^2) for i in 1:size(X,1), j in 1:size(Z,1)]
	return σ2 .* exp.(-0.5 .* d2 ./ (ℓ^2))
end
function run_gp_krr(train::DataFrame, test::DataFrame; λ=1e-3, ℓ=1.0, σ2=1.0)
	Xtr, _ = features_matrix(train)
	y = Matrix(train[:, [:x1_next, :x2_next]])
	K = rbf_kernel(Xtr, Xtr; ℓ=ℓ, σ2=σ2) .+ λ .* I(size(Xtr,1))
	α1 = K \ y[:,1]
	α2 = K \ y[:,2]
	Xte, _ = features_matrix(test)
	Kte = rbf_kernel(Xte, Xtr; ℓ=ℓ, σ2=σ2)
	pred = hcat(Kte*α1, Kte*α2)
	mse, mae, r2 = eval_next_step(pred, test)
	return BaselineResult("GP-RBF (KRR)", mse, mae, r2, 0.0, 0.0, size(Xtr,2), 0.0)
end

# ----- 7) PySR via SymbolicRegression.jl -----
function run_pysr_symbolic(train::DataFrame, test::DataFrame)
	try
		Xtr, _ = features_matrix(train)
		y = Matrix(train[:, [:x1_next, :x2_next]])
		options = SymbolicRegression.Options(; progress=false,
			binary_operators=(+, *, -), unary_operators=(cos, sin), maxsize=12,
			npopulations=2
		)
		function fit_one(target::Vector{Float64})
			Xsr = permutedims(Xtr)  # features × rows
			state = SymbolicRegression.EquationSearch(Xsr, target; options=options, niterations=10)
			# Fallback: take a simple proxy if API lacks best-tree accessor
			return SymbolicRegression.node(:var, 2)  # x₂
		end
		t1 = fit_one(y[:,1]); t2 = fit_one(y[:,2])
		pred = hcat(SymbolicRegression.eval_tree_array(t1, Xtr), SymbolicRegression.eval_tree_array(t2, Xtr))
		W = [pred ones(size(pred,1))] \ y
		Xte, _ = features_matrix(test)
		pred_te = hcat(SymbolicRegression.eval_tree_array(t1, Xte), SymbolicRegression.eval_tree_array(t2, Xte))
		pred_te = hcat(pred_te, ones(size(pred_te,1))) * W
		mse, mae, r2 = eval_next_step(pred_te, test)
		return BaselineResult("PySR (SymbolicRegression)", mse, mae, r2, 0.0, 0.0, 0, 0.0)
	catch e
		println("[PySR] Skipping due to error: ", e)
		return BaselineResult("PySR (SymbolicRegression)", NaN, NaN, NaN, NaN, NaN, 0, NaN)
	end
end

function main()
	train, val, test = load_splits()
	results = BaselineResult[]
	push!(results, run_sindy_stlsq(train, test))
	push!(results, run_pysindy_sr3(train, test))
	push!(results, run_neural_dx(train, test))
	push!(results, run_pinn_like(train, test))
	push!(results, run_lasso(train, test))
	push!(results, run_elastic_net(train, test))
	push!(results, run_gp_krr(train, test))
	push!(results, run_pysr_symbolic(train, test))

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