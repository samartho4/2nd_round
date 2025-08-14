#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Random, Statistics, StatsBase
using CSV, DataFrames, JLD2, Dates, Printf
using HypothesisTests, Distributions
using DifferentialEquations
using BSON

include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
include(joinpath(@__DIR__, "..", "src", "statistical_framework.jl"))

using .Microgrid
using .NeuralNODEArchitectures
using .StatisticalFramework

const N_BOOT = 2000

println("="^70)
println("COMPREHENSIVE STATISTICAL VALIDATION")
println("="^70)

# --- Helpers ---
cohens_d(a::Vector{<:Real}, b::Vector{<:Real}) = begin
	μa, μb = mean(a), mean(b)
	σp = sqrt((var(a) + var(b)) / 2)
	σp == 0 ? 0.0 : (μa - μb) / σp
end

function bonferroni_correct(pvals::Vector{Float64})
	m = length(pvals)
	return clamp.(pvals .* m, 0.0, 1.0)
end

function bootstrap_ci_mean(x::Vector{Float64}; nboot::Int=N_BOOT)
	n = length(x)
	if n == 0
		return (NaN, NaN)
	end
	bmeans = Float64[]
	for _ in 1:nboot
		idx = sample(1:n, n; replace=true)
		push!(bmeans, mean(@view x[idx]))
	end
	return (quantile(bmeans, 0.025), quantile(bmeans, 0.975))
end

# Build score distributions from posterior samples
function trajectory_mse_for_params(deriv_fn::Function, θ::AbstractVector, x0::Vector{Float64}, t::Vector{Float64}; abstol=1e-8, reltol=1e-8)
	prob = ODEProblem(deriv_fn, x0, (t[1], t[end]), collect(θ))
	sol = solve(prob, Tsit5(), saveat=t, abstol=abstol, reltol=reltol, maxiters=10000)
	return sol
end

function main()
	mkpath(joinpath(@__DIR__, "..", "paper", "results"))
	# Load data
	df_test = CSV.read(joinpath(@__DIR__, "..", "data", "test_dataset.csv"), DataFrame)
	first_scn = df_test.scenario[1]
	blk = filter(row -> row.scenario == first_scn, df_test)
	t = Array(blk.time)
	Y = Matrix(blk[:, [:x1, :x2]])
	x0 = Y[1, :]

	# Load results with posterior samples
	bnn_res = BSON.load(joinpath(@__DIR__, "..", "checkpoints", "bayesian_neural_ode_results.bson"))[:bayesian_results]
	ude_res = BSON.load(joinpath(@__DIR__, "..", "checkpoints", "ude_results_fixed.bson"))[:ude_results]

	# Architecture for BNN
	arch_name = haskey(bnn_res, :arch) ? String(bnn_res[:arch]) : "baseline"
	function pick_arch(arch::AbstractString)
		a = lowercase(String(arch))
		if a == "baseline"
			return (:baseline, baseline_nn!, 10)
		elseif a == "baseline_bias"
			return (:baseline_bias, baseline_nn_bias!, 14)
		elseif a == "deep"
			return (:deep, deep_nn!, 26)
		else
			return (:baseline, baseline_nn!, 10)
		end
	end
	arch_sym, bayes_deriv_fn, _ = pick_arch(arch_name)

	# Build score arrays from posterior samples
	bnn_params_samples = get(bnn_res, :param_samples, nothing)
	ude_phys = get(ude_res, :physics_samples, nothing)
	ude_nn = get(ude_res, :neural_samples, nothing)

	bnn_scores = Float64[]
	ude_scores = Float64[]
	phys_scores = Float64[]

	# Observational truth matrix at t
	truth = Y

	# BNN
	if bnn_params_samples !== nothing
		for θ in eachrow(bnn_params_samples)
			try
				sol = trajectory_mse_for_params(bayes_deriv_fn, θ, x0, t)
				pred = hcat(sol.u...)'
				push!(bnn_scores, mean((pred .- truth).^2))
			catch; end
		end
	end

	# UDE
	if ude_phys !== nothing && ude_nn !== nothing
		ns = min(size(ude_phys,1), size(ude_nn,1))
		function ude_dyn!(dx,x,p,t)
			x1, x2 = x
			ηin, ηout, α, β, γ = p[1:5]
			nn_params = p[6:end]
			u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
			Pgen = max(0, sin((t - 6) * π / 12))
			Pload = 0.6 + 0.2 * sin(t * π / 12)
			Pin = u > 0 ? ηin * u : (1 / ηout) * u
			dx[1] = Pin - Pload
			nn_output = NeuralNODEArchitectures.ude_nn_forward(x1, x2, Pgen, Pload, t, nn_params)
			dx[2] = -α * x2 + nn_output + γ * x1
		end
		for k in 1:ns
			p = [ude_phys[k, :]..., ude_nn[k, :]...]
			try
				prob = ODEProblem(ude_dyn!, x0, (t[1], t[end]), p)
				sol = solve(prob, Tsit5(), saveat=t, abstol=1e-8, reltol=1e-8, maxiters=10000)
				pred = hcat(sol.u...)'
				push!(ude_scores, mean((pred .- truth).^2))
			catch; end
		end
	end

	# Physics-only single score
	try
		prob = ODEProblem(Microgrid.microgrid!, x0, (t[1], t[end]), (0.9,0.9,0.3,1.2,0.4))
		sol = solve(prob, Tsit5(), saveat=t, abstol=1e-8, reltol=1e-8, maxiters=10000)
		pred = hcat(sol.u...)'
		push!(phys_scores, mean((pred .- truth).^2))
	catch; end

	# Summaries
	method_scores = Dict{String,Vector{Float64}}()
	if !isempty(bnn_scores); method_scores["BNN_ODE"] = bnn_scores; end
	if !isempty(ude_scores); method_scores["UDE"] = ude_scores; end
	if !isempty(phys_scores); method_scores["Physics-Only"] = phys_scores; end

	method_stats = Dict{String, StatisticalFramework.StatisticalResults}()
	for (m, v) in method_scores
		method_stats[m] = StatisticalFramework.compute_statistical_summary(v)
	end

	# Pairwise significance and effect sizes
	comparisons = DataFrame(Comparison=String[], p_value=Float64[], bonferroni_p=Float64[], d=Float64[])
	methods = collect(keys(method_scores))
	for i in 1:length(methods), j in i+1:length(methods)
		ma, mb = methods[i], methods[j]
		a, b = method_scores[ma], method_scores[mb]
		if length(a) >= 3 && length(b) >= 3 && std(a) > 0 && std(b) > 0
			p = try
				pvalue(UnequalVarianceTTest(a, b))
			catch
				NaN
			end
			d = cohens_d(a,b)
			push!(comparisons, ("$ma vs $mb", p, bonferroni_correct([p])[1], d))
		end
	end

	# Write markdown
	out = joinpath(@__DIR__, "..", "paper", "results", "enhanced_stats_summary.md")
	open(out, "w") do f
		println(f, "# Statistical Validation Summary")
		println(f, "\n## Score distributions (trajectory MSE)")
		println(f, "\n| Method | Mean±Std | 95% CI | n |")
		println(f, "|---|---|---|---|")
		for (name, s) in sort(collect(method_stats); by=x->x[1])
			μ = @sprintf("%.4f", s.mean)
			σ = @sprintf("%.4f", s.std)
			ci = @sprintf("[%.4f, %.4f]", s.ci_lower, s.ci_upper)
			println(f, "| $(name) | $(μ)±$(σ) | $(ci) | $(s.n_samples) |")
		end
		println(f, "\n## Significance tests")
		println(f, "\n| Comparison | p-value | Bonferroni p | Cohen's d |")
		println(f, "|---|---|---|---|")
		for r in eachrow(comparisons)
			println(f, "| $(r.Comparison) | $( @sprintf("%.4g", r.p_value) ) | $( @sprintf("%.4g", r.bonferroni_p) ) | $( @sprintf("%.3f", r.d) ) |")
		end
	end
	println("✅ Saved paper/results/enhanced_stats_summary.md")
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end 