#!/usr/bin/env julia

"""
    evaluate_full_retraining.jl

Paper-ready evaluation for UDE and BNode models trained on the full dataset.
- Loads parameters from checkpoints/comprehensive_full_dataset_results.bson
- Evaluates on training, validation, and test datasets
- Computes RMSE, MAE, and R² for x1 and x2
- Writes results to results/paper_evaluation_summary.md
"""

using Pkg
Pkg.activate(".")

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DifferentialEquations
using Statistics
using CSV
using DataFrames
using BSON
using Dates

println("🧪 PAPER EVALUATION: UDE vs BNode (posterior mean)")
println("=" ^ 60)

# -----------------------------------------------------------------------------
# Load saved training results
# -----------------------------------------------------------------------------
ckpt_path = joinpath(@__DIR__, "..", "checkpoints", "comprehensive_full_dataset_results.bson")
if !isfile(ckpt_path)
	error("Checkpoint not found: $(ckpt_path). Please run retraining first.")
end

println("📦 Loading checkpoint: $(ckpt_path)")
BSON.@load ckpt_path comprehensive_results

ude_params = vcat(
	comprehensive_results[:ude_results][:physics_params][:ηin],
	comprehensive_results[:ude_results][:physics_params][:ηout],
	comprehensive_results[:ude_results][:physics_params][:α],
	comprehensive_results[:ude_results][:physics_params][:β],
	comprehensive_results[:ude_results][:physics_params][:γ],
	comprehensive_results[:ude_results][:neural_params],
)

bnode_posterior_mean = Vector{Float64}(comprehensive_results[:bnode_results][:posterior_mean])

println("  → Loaded UDE params: $(length(ude_params)) dims")
println("  → Loaded BNode posterior mean: $(length(bnode_posterior_mean)) dims")

# -----------------------------------------------------------------------------
# Model definitions (must match training script equations)
# -----------------------------------------------------------------------------
function nn_forward(x::NTuple{2,Float64}, params::AbstractVector{<:Real})
	# params[6:end]: 15 weights for 2->5 tanh -> sum
	x1, x2 = x
	nn_weights = params[6:end]
	W1 = reshape(nn_weights[1:10], 5, 2)
	b1 = nn_weights[11:15]
	inputs = [x1, x2]
	h = tanh.(W1 * inputs .+ b1)
	return sum(h)
end

function ude_system!(du, u, p, t)
	x1, x2 = u
	ηin, ηout, α, β, γ = p[1:5]
	neural_correction = nn_forward((x1, x2), p)
	# Roadmap alignment for evaluation (matches training): physics-only Eq1; NN only in Eq2
	du[1] = ηin * max(0, x2) - (1/ηout) * max(0, -x2) - α * x1
	du[2] = -α * x2 + γ * x1 + neural_correction
end

# -----------------------------------------------------------------------------
# Utility: Solve ODE at dataset times and return predictions
# -----------------------------------------------------------------------------
function predict_times(params::AbstractVector{<:Real}, t_vec::Vector{Float64})
	prob = ODEProblem(ude_system!, [0.5, 0.0], (minimum(t_vec), maximum(t_vec)), params)
	sol = solve(prob, Tsit5(); saveat=t_vec, abstol=1e-6, reltol=1e-6)
	if sol.retcode != :Success
		return nothing
	end
	Y_pred = reduce(hcat, (sol(t) for t in t_vec))'
	return Matrix(Y_pred)
end

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
struct Metrics
	rmse_x1::Float64
	rmse_x2::Float64
	r2_x1::Float64
	r2_x2::Float64
	mae_x1::Float64
	mae_x2::Float64
end

function compute_metrics(Y_true::Matrix{Float64}, Y_pred::Matrix{Float64})
	rmse_x1 = sqrt(mean((Y_pred[:, 1] .- Y_true[:, 1]).^2))
	rmse_x2 = sqrt(mean((Y_pred[:, 2] .- Y_true[:, 2]).^2))
	r2_x1 = 1 - sum((Y_pred[:, 1] .- Y_true[:, 1]).^2) / sum((Y_true[:, 1] .- mean(Y_true[:, 1])).^2)
	r2_x2 = 1 - sum((Y_pred[:, 2] .- Y_true[:, 2]).^2) / sum((Y_true[:, 2] .- mean(Y_true[:, 2])).^2)
	mae_x1 = mean(abs.(Y_pred[:, 1] .- Y_true[:, 1]))
	mae_x2 = mean(abs.(Y_pred[:, 2] .- Y_true[:, 2]))
	return Metrics(rmse_x1, rmse_x2, r2_x1, r2_x2, mae_x1, mae_x2)
end

function eval_split(split_name::String, csv_path::String)
	println("\n📄 Evaluating split: $(split_name)")
	if !isfile(csv_path)
		println("  → Missing file, skipping: $(csv_path)")
		return nothing
	end
	df = CSV.read(csv_path, DataFrame)
	t = Vector{Float64}(df.time)
	Y = Matrix(df[:, [:x1, :x2]])

	# UDE predictions
	Y_pred_ude = predict_times(ude_params, t)
	if Y_pred_ude === nothing
		println("  → UDE solve failed on $(split_name)")
	else
		m_ude = compute_metrics(Y, Y_pred_ude)
		println("  UDE:  RMSE[x1,x2]=($(round(m_ude.rmse_x1, digits=6)), $(round(m_ude.rmse_x2, digits=6)))  R²[x1,x2]=($(round(m_ude.r2_x1, digits=6)), $(round(m_ude.r2_x2, digits=6)))  MAE[x1,x2]=($(round(m_ude.mae_x1, digits=6)), $(round(m_ude.mae_x2, digits=6)))")
	end

	# BNode posterior mean predictions
	Y_pred_bnode = predict_times(bnode_posterior_mean, t)
	if Y_pred_bnode === nothing
		println("  → BNode solve failed on $(split_name)")
	else
		m_b = compute_metrics(Y, Y_pred_bnode)
		println("  BNode: RMSE[x1,x2]=($(round(m_b.rmse_x1, digits=6)), $(round(m_b.rmse_x2, digits=6)))  R²[x1,x2]=($(round(m_b.r2_x1, digits=6)), $(round(m_b.r2_x2, digits=6)))  MAE[x1,x2]=($(round(m_b.mae_x1, digits=6)), $(round(m_b.mae_x2, digits=6)))")
	end

	return (; t, Y, Y_pred_ude, Y_pred_bnode)
end

# -----------------------------------------------------------------------------
# Run evaluation for each split
# -----------------------------------------------------------------------------
train_csv = joinpath(@__DIR__, "..", "data", "training_dataset_fixed.csv")
val_csv   = joinpath(@__DIR__, "..", "data", "validation_dataset_fixed.csv")
test_csv  = joinpath(@__DIR__, "..", "data", "test_dataset_fixed.csv")

res_train = eval_split("train", train_csv)
res_val   = eval_split("val",   val_csv)
res_test  = eval_split("test",  test_csv)

# -----------------------------------------------------------------------------
# Persist a paper-ready summary
# -----------------------------------------------------------------------------
results_dir = joinpath(@__DIR__, "..", "results")
if !isdir(results_dir)
	mkdir(results_dir)
end

summary_md = joinpath(results_dir, "paper_evaluation_summary.md")
open(summary_md, "w") do io
	write(io, "# Paper Evaluation Summary\n\n")
	write(io, "Date: $(now())\n\n")
	write(io, "Datasets evaluated: train, val, test (full). Metrics: RMSE, MAE, R² per variable.\n\n")
	for (name, res) in [("train", res_train), ("val", res_val), ("test", res_test)]
		if res === nothing
			continue
		end
		Y = res[:Y]
		if res[:Y_pred_ude] !== nothing
			m = compute_metrics(Y, res[:Y_pred_ude])
			write(io, "## $(name) — UDE\n")
			write(io, "- RMSE: x1=$(round(m.rmse_x1, digits=6)), x2=$(round(m.rmse_x2, digits=6))\n")
			write(io, "- MAE:  x1=$(round(m.mae_x1, digits=6)), x2=$(round(m.mae_x2, digits=6))\n")
			write(io, "- R²:   x1=$(round(m.r2_x1, digits=6)), x2=$(round(m.r2_x2, digits=6))\n\n")
		end
		if res[:Y_pred_bnode] !== nothing
			m = compute_metrics(Y, res[:Y_pred_bnode])
			write(io, "## $(name) — BNode (posterior mean)\n")
			write(io, "- RMSE: x1=$(round(m.rmse_x1, digits=6)), x2=$(round(m.rmse_x2, digits=6))\n")
			write(io, "- MAE:  x1=$(round(m.mae_x1, digits=6)), x2=$(round(m.mae_x2, digits=6))\n")
			write(io, "- R²:   x1=$(round(m.r2_x1, digits=6)), x2=$(round(m.r2_x2, digits=6))\n\n")
		end
	end
end

println("\n📝 Saved paper evaluation summary → $(summary_md)")
println("✅ Evaluation completed") 