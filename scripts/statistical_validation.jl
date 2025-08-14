#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Random, Statistics, StatsBase
using CSV, DataFrames, JLD2, Dates, Printf
using HypothesisTests, Distributions
using DifferentialEquations

include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "statistical_framework.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline_models.jl"))
include(joinpath(@__DIR__, "..", "src", "statistical_evaluation.jl"))

using .Microgrid
using .StatisticalFramework
using .BaselineModels

const N_BOOT = 2000
const N_SEEDS = 10
const SEEDS = 42:(42+N_SEEDS-1)

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

function power_analysis_ttest(effect_d::Float64; α=0.05, power=0.8)
	# Approximate sample size per group for two-sample t-test
	# n ≈ 2 * (z_{1-α/2} + z_{power})^2 / d^2
	zα = quantile(Normal(), 1 - α/2)
	zβ = quantile(Normal(), power)
	return ceil(Int, 2 * (zα + zβ)^2 / max(1e-12, effect_d^2))
end

function paired_tests(a::Vector{Float64}, b::Vector{Float64})
	@assert length(a) == length(b)
	d = a .- b
	# Paired t-test
	tres = OneSampleTTest(d, 0.0)
	p_t = pvalue(tres)
	# Wilcoxon signed-rank (approx; requires nonzero diffs)
	nd = filter(!=(0.0), d)
	p_w = length(nd) > 0 ? pvalue(SignedRankTest(nd)) : NaN
	return p_t, p_w
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

# --- Load per-seed scores if available ---
function load_seed_scores()
	# Expected files from training runs
	res = Dict{String,Vector{Float64}}(
		"UDE" => Float64[],
		"BNN_ODE" => Float64[],
		"Physics-Only" => Float64[]
	)
	for s in SEEDS
		# Try canonical paths; skip if missing
		ude_path = joinpath(@__DIR__, "..", "checkpoints", "ude_seed_$(s).jld2")
		bnn_path = joinpath(@__DIR__, "..", "checkpoints", "bnn_ode_seed_$(s).jld2")
		if isfile(ude_path)
			try
				ud = JLD2.load(ude_path)
				mse = haskey(ud, "mse") ? ud["mse"] : NaN
				if !isnan(mse); push!(res["UDE"], mse); end
			catch; end
		end
		if isfile(bnn_path)
			try
				bd = JLD2.load(bnn_path)
				mse = haskey(bd, "mse") ? bd["mse"] : NaN
				if !isnan(mse); push!(res["BNN_ODE"], mse); end
			catch; end
		end
	end
	return res
end

# Fallback: compute quick physics-only baseline MSE from data
function compute_physics_only_mse()
	try
		df = CSV.read(joinpath(@__DIR__, "..", "data", "test_dataset.csv"), DataFrame)
		t = Array(df.time)
		Y = Matrix(df[:, [:x1, :x2]])
		x0 = Y[1, :]
		prob = ODEProblem(Microgrid.microgrid!, x0, (t[1], t[end]), (0.9,0.9,0.3,1.2,0.4))
		sol = solve(prob, Tsit5(), saveat=t)
		Yp = hcat(sol.u...)'
		return mean((Yp .- Y).^2)
	catch
		return NaN
	end
end

# --- Main ---
function main()
	mkpath(joinpath(@__DIR__, "..", "paper", "results"))
	println("Loading per-seed results...")
	scores = load_seed_scores()
	if isempty(scores["Physics-Only"]) 
		ph = compute_physics_only_mse()
		if !isnan(ph)
			scores["Physics-Only"] = fill(ph, N_SEEDS)
		end
	end

	# Summaries
	summ = Dict{String,Any}()
	for (m, v) in scores
		if isempty(v); continue; end
		ci = bootstrap_ci_mean(v)
		summ[m] = Dict(
			"mean"=>mean(v), "std"=>std(v), "ci"=>ci, "n"=>length(v)
		)
	end

	# Pairwise significance, effect sizes
	methods = collect(keys(scores))
	pairs = [(a,b) for (i,a) in enumerate(methods) for b in methods[(i+1):end]]
	pvals = Float64[]
	rows = DataFrame(MethodA=String[], MethodB=String[], p_t=Float64[], p_w=Float64[], d=Float64[], nA=Int[], nB=Int[])
	for (a,b) in pairs
		va, vb = scores[a], scores[b]
		if isempty(va) || isempty(vb); continue; end
		pt, pw = paired_tests(va[1:min(end,end)], vb[1:min(end,end)])
		d = cohens_d(va, vb)
		push!(pvals, pt)
		push!(rows, (a,b,pt,pw,d,length(va),length(vb)))
	end
	if !isempty(pvals)
		rows.p_t_bonf = bonferroni_correct(copy(pvals))
	end

	# Power analysis for UDE vs BNN_ODE
	if all(haskey.(Ref(summ), ["UDE","BNN_ODE"]))
		d_est = abs(cohens_d(scores["UDE"], scores["BNN_ODE"]))
		n_req = power_analysis_ttest(d_est)
		println(@sprintf("Estimated effect size d=%.3f; required n per group for 80%% power ≈ %d", d_est, n_req))
	end

	# Save table
	open(joinpath(@__DIR__, "..", "paper", "results", "enhanced_stats_summary.md"), "w") do f
		println(f, "| Method | Mean±Std | 95% CI | n |")
		println(f, "|---|---|---|---|")
		for m in sort!(collect(keys(summ)))
			S = summ[m]
			println(f, @sprintf("| %s | %.3f±%.3f | [%.3f, %.3f] | %d |", m, S["mean"], S["std"], S["ci"][1], S["ci"][2], S["n"]))
		end
		println(f, "\n| Comparison | p_t | p_wilcoxon | d | Bonferroni p_t |")
		println(f, "|---|---|---|---|---|")
		for i in 1:nrow(rows)
			println(f, @sprintf("| %s vs %s | %.4g | %.4g | %.3f | %.4g |", rows.MethodA[i], rows.MethodB[i], rows.p_t[i], rows.p_w[i], rows.d[i], get(rows, :p_t_bonf, fill(NaN, nrow(rows)))[i]))
		end
	end
	println("✅ Saved paper/results/enhanced_stats_summary.md")
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end 