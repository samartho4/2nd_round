#!/usr/bin/env julia

"""
    proper_evaluation.jl

Proper evaluation methodology that:
1. Evaluates ALL test scenarios (not just one)
2. Provides statistically valid results
3. Uses correct evaluation metrics
4. Includes confidence intervals and significance testing
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, BSON, LinearAlgebra
using Random, Printf, HypothesisTests, Distributions

println("🔬 PROPER EVALUATION METHODOLOGY")
println("=" ^ 60)

# Set random seed for reproducibility
Random.seed!(42)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

println("\n📊 LOADING DATA")
println("-" ^ 30)

# Load all datasets
test_data = CSV.read("data/test_dataset.csv", DataFrame)
train_data = CSV.read("data/training_dataset.csv", DataFrame)
val_data = CSV.read("data/validation_dataset.csv", DataFrame)

println("Test data: $(nrow(test_data)) samples")
println("Training data: $(nrow(train_data)) samples")
println("Validation data: $(nrow(val_data)) samples")

# Get unique scenarios
test_scenarios = unique(test_data.scenario)
train_scenarios = unique(train_data.scenario)
val_scenarios = unique(val_data.scenario)

println("Test scenarios: $(length(test_scenarios))")
println("Training scenarios: $(length(train_scenarios))")
println("Validation scenarios: $(length(val_scenarios))")

# Check for data leakage
overlap = intersect(train_scenarios, test_scenarios)
if length(overlap) > 0
    println("⚠️  WARNING: Data leakage detected!")
    println("Overlapping scenarios: $overlap")
else
    println("✅ No data leakage detected")
end

# ============================================================================
# LOAD MODELS
# ============================================================================

println("\n🤖 LOADING MODELS")
println("-" ^ 30)

# Load BNN-ODE model
bnn_path = "checkpoints/bayesian_neural_ode_results.bson"
if isfile(bnn_path)
    println("✅ Loading BNN-ODE model...")
    bnn_data = BSON.load(bnn_path)
    bnn_results = bnn_data[:bayesian_results]
    bnn_params = bnn_results[:params_mean]
    println("  → Parameters: $(length(bnn_params))")
else
    println("❌ BNN-ODE model not found")
    exit(1)
end

# Load UDE model
ude_path = "checkpoints/ude_results_fixed.bson"
if isfile(ude_path)
    println("✅ Loading UDE model...")
    ude_data = BSON.load(ude_path)
    ude_results = ude_data[:ude_results]
    physics_params = ude_results[:physics_params_mean]
    neural_params = ude_results[:neural_params_mean]
    println("  → Physics parameters: $(length(physics_params))")
    println("  → Neural parameters: $(length(neural_params))")
else
    println("❌ UDE model not found")
    exit(1)
end

# ============================================================================
# DEFINE MODEL DYNAMICS FUNCTIONS
# ============================================================================

println("\n🔧 DEFINING MODEL DYNAMICS")
println("-" ^ 30)

# BNN-ODE dynamics function
function bnn_dynamics!(dx, x, p, t)
    x1, x2 = x
    h1 = tanh(p[1]*x1 + p[2]*x2 + p[3]*t + p[4])
    h2 = tanh(p[5]*x1 + p[6]*x2 + p[7]*t + p[8])
    dx[1] = p[13] * h1
    dx[2] = p[14] * h2
end

# UDE dynamics function
function ude_dynamics!(dx, x, p, t)
    x1, x2 = x
    ηin, ηout, α, β, γ = p[1:5]
    nn_params = p[6:end]
    
    # Control input and generation/load
    u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
    Pgen = max(0, sin((t - 6) * π / 12))
    Pload = 0.6 + 0.2 * sin(t * π / 12)
    
    # Physics-based terms
    Pin = u > 0 ? ηin * u : (1 / ηout) * u
    dx[1] = Pin - Pload
    
    # Neural network for nonlinear term
    h1 = tanh(nn_params[1]*x1 + nn_params[2]*x2 + nn_params[3]*Pgen + nn_params[4]*Pload + nn_params[5]*t + nn_params[6])
    h2 = tanh(nn_params[7]*x1 + nn_params[8]*x2 + nn_params[9]*Pgen + nn_params[10]*Pload + nn_params[11]*t + nn_params[12])
    nn_output = nn_params[13]*h1 + nn_params[14]*h2 + nn_params[15]
    
    dx[2] = -α * x2 + nn_output + γ * x1
end

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

function evaluate_scenario(scenario_data, bnn_params, ude_params)
    """Evaluate a single scenario and return metrics"""
    
    # Sort by time
    sort!(scenario_data, :time)
    
    # Prepare data
    t_data = Array(scenario_data.time)
    Y_data = Matrix(scenario_data[:, [:x1, :x2]])
    
    # Calculate actual derivatives
    actual_derivatives = diff(Y_data, dims=1) ./ diff(t_data)
    t_derivatives = t_data[1:end-1]
    
    # Make predictions
    bnn_predictions = []
    ude_predictions = []
    
    for i in 1:length(t_derivatives)
        x = Y_data[i, :]
        t = t_derivatives[i]
        
        # BNN-ODE prediction
        dx_bnn = zeros(2)
        bnn_dynamics!(dx_bnn, x, bnn_params, t)
        push!(bnn_predictions, dx_bnn)
        
        # UDE prediction
        dx_ude = zeros(2)
        ude_dynamics!(dx_ude, x, ude_params, t)
        push!(ude_predictions, dx_ude)
    end
    
    bnn_predictions = hcat(bnn_predictions...)'
    ude_predictions = hcat(ude_predictions...)'
    
    # Calculate errors
    bnn_errors = bnn_predictions .- actual_derivatives
    ude_errors = ude_predictions .- actual_derivatives
    
    # Compute metrics
    bnn_metrics = Dict(
        "mse_x1" => mean(bnn_errors[:, 1].^2),
        "mse_x2" => mean(bnn_errors[:, 2].^2),
        "mse_total" => mean(bnn_errors.^2),
        "rmse_x1" => sqrt(mean(bnn_errors[:, 1].^2)),
        "rmse_x2" => sqrt(mean(bnn_errors[:, 2].^2)),
        "rmse_total" => sqrt(mean(bnn_errors.^2)),
        "mae_x1" => mean(abs.(bnn_errors[:, 1])),
        "mae_x2" => mean(abs.(bnn_errors[:, 2])),
        "mae_total" => mean(abs.(bnn_errors))
    )
    
    ude_metrics = Dict(
        "mse_x1" => mean(ude_errors[:, 1].^2),
        "mse_x2" => mean(ude_errors[:, 2].^2),
        "mse_total" => mean(ude_errors.^2),
        "rmse_x1" => sqrt(mean(ude_errors[:, 1].^2)),
        "rmse_x2" => sqrt(mean(ude_errors[:, 2].^2)),
        "rmse_total" => sqrt(mean(ude_errors.^2)),
        "mae_x1" => mean(abs.(ude_errors[:, 1])),
        "mae_x2" => mean(abs.(ude_errors[:, 2])),
        "mae_total" => mean(abs.(ude_errors))
    )
    
    return bnn_metrics, ude_metrics
end

# ============================================================================
# EVALUATE ALL SCENARIOS
# ============================================================================

println("\n📈 EVALUATING ALL SCENARIOS")
println("-" ^ 30)

# Prepare UDE parameters
ude_params = [physics_params..., neural_params...]

# Store results for all scenarios
all_results = []

for (i, scenario) in enumerate(test_scenarios)
    println("Evaluating scenario $i/$(length(test_scenarios)): $scenario")
    
    # Get scenario data
    scenario_data = test_data[test_data.scenario .== scenario, :]
    
    if nrow(scenario_data) < 2
        println("  ⚠️  Skipping scenario with insufficient data")
        continue
    end
    
    # Evaluate scenario
    bnn_metrics, ude_metrics = evaluate_scenario(scenario_data, bnn_params, ude_params)
    
    # Store results
    push!(all_results, Dict(
        "scenario" => scenario,
        "samples" => nrow(scenario_data),
        "bnn" => bnn_metrics,
        "ude" => ude_metrics
    ))
end

println("✅ Evaluated $(length(all_results)) scenarios")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

println("\n📊 STATISTICAL ANALYSIS")
println("-" ^ 30)

# Extract metrics for statistical testing
bnn_mse_x1 = [r["bnn"]["mse_x1"] for r in all_results]
bnn_mse_x2 = [r["bnn"]["mse_x2"] for r in all_results]
bnn_mse_total = [r["bnn"]["mse_total"] for r in all_results]

ude_mse_x1 = [r["ude"]["mse_x1"] for r in all_results]
ude_mse_x2 = [r["ude"]["mse_x2"] for r in all_results]
ude_mse_total = [r["ude"]["mse_total"] for r in all_results]

# Statistical tests
println("Statistical significance testing (Wilcoxon signed-rank test):")

# MSE x1 comparison
test_x1 = SignedRankTest(bnn_mse_x1, ude_mse_x1)
println("MSE x1: p-value = $(pvalue(test_x1))")

# MSE x2 comparison  
test_x2 = SignedRankTest(bnn_mse_x2, ude_mse_x2)
println("MSE x2: p-value = $(pvalue(test_x2))")

# MSE total comparison
test_total = SignedRankTest(bnn_mse_total, ude_mse_total)
println("MSE total: p-value = $(pvalue(test_total))")

# Effect sizes (Cohen's d)
function cohens_d(x, y)
    n1, n2 = length(x), length(y)
    pooled_std = sqrt(((n1-1)*var(x) + (n2-1)*var(y)) / (n1 + n2 - 2))
    return (mean(x) - mean(y)) / pooled_std
end

println("\nEffect sizes (Cohen's d):")
println("MSE x1: $(cohens_d(bnn_mse_x1, ude_mse_x1))")
println("MSE x2: $(cohens_d(bnn_mse_x2, ude_mse_x2))")
println("MSE total: $(cohens_d(bnn_mse_total, ude_mse_total))")

# Confidence intervals
function confidence_interval(x, confidence=0.95)
    n = length(x)
    t_val = quantile(TDist(n-1), (1 + confidence) / 2)
    se = std(x) / sqrt(n)
    margin = t_val * se
    return (mean(x) - margin, mean(x) + margin)
end

println("\n95% Confidence Intervals:")
println("BNN-ODE MSE x1: $(confidence_interval(bnn_mse_x1))")
println("UDE MSE x1: $(confidence_interval(ude_mse_x1))")
println("BNN-ODE MSE x2: $(confidence_interval(bnn_mse_x2))")
println("UDE MSE x2: $(confidence_interval(ude_mse_x2))")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

println("\n📋 SUMMARY STATISTICS")
println("-" ^ 30)

# Overall means
bnn_mean_x1 = mean(bnn_mse_x1)
bnn_mean_x2 = mean(bnn_mse_x2)
bnn_mean_total = mean(bnn_mse_total)

ude_mean_x1 = mean(ude_mse_x1)
ude_mean_x2 = mean(ude_mse_x2)
ude_mean_total = mean(ude_mse_total)

println("BNN-ODE Performance:")
println("  MSE x1: $(bnn_mean_x1) ± $(std(bnn_mse_x1))")
println("  MSE x2: $(bnn_mean_x2) ± $(std(bnn_mse_x2))")
println("  MSE total: $(bnn_mean_total) ± $(std(bnn_mse_total))")

println("\nUDE Performance:")
println("  MSE x1: $(ude_mean_x1) ± $(std(ude_mse_x1))")
println("  MSE x2: $(ude_mean_x2) ± $(std(ude_mse_x2))")
println("  MSE total: $(ude_mean_total) ± $(std(ude_mse_total))")

# Performance ratios
println("\nPerformance Ratios:")
println("MSE x1 ratio (BNN-ODE/UDE): $(bnn_mean_x1/ude_mean_x1)")
println("MSE x2 ratio (BNN-ODE/UDE): $(bnn_mean_x2/ude_mean_x2)")
println("MSE total ratio (BNN-ODE/UDE): $(bnn_mean_total/ude_mean_total)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

println("\n💾 SAVING RESULTS")
println("-" ^ 30)

# Create comprehensive results DataFrame
results_data = []
for result in all_results
    push!(results_data, Dict(
        "scenario" => result["scenario"],
        "samples" => result["samples"],
        "bnn_mse_x1" => result["bnn"]["mse_x1"],
        "bnn_mse_x2" => result["bnn"]["mse_x2"],
        "bnn_mse_total" => result["bnn"]["mse_total"],
        "ude_mse_x1" => result["ude"]["mse_x1"],
        "ude_mse_x2" => result["ude"]["mse_x2"],
        "ude_mse_total" => result["ude"]["mse_total"]
    ))
end

results_df = DataFrame(results_data)
CSV.write("results/comprehensive_evaluation_results.csv", results_df)

# Save summary statistics
summary_stats = Dict(
    "evaluation_date" => "2025-08-17",
    "random_seed" => 42,
    "total_scenarios" => length(all_results),
    "bnn_mean_mse_x1" => bnn_mean_x1,
    "bnn_std_mse_x1" => std(bnn_mse_x1),
    "bnn_mean_mse_x2" => bnn_mean_x2,
    "bnn_std_mse_x2" => std(bnn_mse_x2),
    "bnn_mean_mse_total" => bnn_mean_total,
    "bnn_std_mse_total" => std(bnn_mse_total),
    "ude_mean_mse_x1" => ude_mean_x1,
    "ude_std_mse_x1" => std(ude_mse_x1),
    "ude_mean_mse_x2" => ude_mean_x2,
    "ude_std_mse_x2" => std(ude_mse_x2),
    "ude_mean_mse_total" => ude_mean_total,
    "ude_std_mse_total" => std(ude_mse_total),
    "p_value_x1" => pvalue(test_x1),
    "p_value_x2" => pvalue(test_x2),
    "p_value_total" => pvalue(test_total),
    "effect_size_x1" => cohens_d(bnn_mse_x1, ude_mse_x1),
    "effect_size_x2" => cohens_d(bnn_mse_x2, ude_mse_x2),
    "effect_size_total" => cohens_d(bnn_mse_total, ude_mse_total)
)

using TOML
open("results/evaluation_summary.toml", "w") do io
    TOML.print(io, summary_stats)
end

println("✅ Results saved to:")
println("  - results/comprehensive_evaluation_results.csv")
println("  - results/evaluation_summary.toml")

# ============================================================================
# FINAL REPORT
# ============================================================================

println("\n📊 FINAL EVALUATION REPORT")
println("-" ^ 30)

println("EVALUATION METHODOLOGY:")
println("✅ Evaluated ALL $(length(all_results)) test scenarios")
println("✅ Used proper statistical testing (Wilcoxon signed-rank)")
println("✅ Computed effect sizes (Cohen's d)")
println("✅ Provided 95% confidence intervals")
println("✅ Set random seed for reproducibility")

println("\nSTATISTICAL SIGNIFICANCE:")
println("MSE x1: p = $(pvalue(test_x1)) $(pvalue(test_x1) < 0.05 ? "SIGNIFICANT" : "NOT SIGNIFICANT")")
println("MSE x2: p = $(pvalue(test_x2)) $(pvalue(test_x2) < 0.05 ? "SIGNIFICANT" : "NOT SIGNIFICANT")")
println("MSE total: p = $(pvalue(test_total)) $(pvalue(test_total) < 0.05 ? "SIGNIFICANT" : "NOT SIGNIFICANT")")

println("\nPERFORMANCE COMPARISON:")
if bnn_mean_total < ude_mean_total
    println("BNN-ODE performs better overall (lower MSE)")
else
    println("UDE performs better overall (lower MSE)")
end

println("\n" ^ 60)
println("🔬 PROPER EVALUATION COMPLETE")
println("=" ^ 60) 