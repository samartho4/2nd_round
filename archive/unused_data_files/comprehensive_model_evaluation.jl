#!/usr/bin/env julia

"""
    comprehensive_model_evaluation.jl

Comprehensive model evaluation comparing BNN-ODE and UDE models with expanded data.
Evaluates using standard metrics: MSE, RMSE, MAE, R² for both state variables.
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, BSON, LinearAlgebra
using Random, Printf

println("🔬 COMPREHENSIVE MODEL EVALUATION")
println("=" ^ 60)

# Load test data
println("📊 Loading test data...")
test_data = CSV.read("data/test_dataset.csv", DataFrame)
println("  → Test samples: $(nrow(test_data))")
println("  → Time range: $(minimum(test_data.time)) - $(maximum(test_data.time)) hours")

# Prepare test data
t_test = Array(test_data.time)
Y_test = Matrix(test_data[:, [:x1, :x2]])
println("  → State variables: x1 (SOC), x2 (Power)")
println("  → Data shape: $(size(Y_test))")

# Calculate actual derivatives for comparison
println("\n📈 Computing actual derivatives...")
actual_derivatives = diff(Y_test, dims=1) ./ diff(t_test)
t_derivatives = t_test[1:end-1]
println("  → Derivatives computed: $(size(actual_derivatives, 1)) points")

# Check for any issues in actual derivatives
println("  → Actual derivatives range: x1 ($(minimum(actual_derivatives[:, 1])), $(maximum(actual_derivatives[:, 1]))), x2 ($(minimum(actual_derivatives[:, 2])), $(maximum(actual_derivatives[:, 2])))")

# Initialize results storage
results = Dict{String, Dict{String, Float64}}()

# ============================================================================
# BAYESIAN NEURAL ODE EVALUATION
# ============================================================================

println("\n" * "="^60)
println("BAYESIAN NEURAL ODE EVALUATION")
println("="^60)

# Load BNN-ODE model
bnn_path = "checkpoints/bayesian_neural_ode_results.bson"
if isfile(bnn_path)
    println("✅ Loading BNN-ODE model...")
    bnn_data = BSON.load(bnn_path)
    bnn_results = bnn_data[:bayesian_results]
    
    println("  → Architecture: $(bnn_results[:arch])")
    println("  → Parameters: $(length(bnn_results[:params_mean]))")
    println("  → MCMC samples: $(bnn_results[:n_samples])")
    
    # Get model parameters
    bnn_params = bnn_results[:params_mean]
    println("  → Parameter range: ($(minimum(bnn_params)), $(maximum(bnn_params)))")
    
    # Define BNN-ODE dynamics function (baseline_bias architecture)
    function bnn_dynamics!(dx, x, p, t)
        x1, x2 = x
        
        # Baseline bias architecture with 14 parameters
        # Input layer: [x1, x2, t] -> hidden layer -> output layer
        h1 = tanh(p[1]*x1 + p[2]*x2 + p[3]*t + p[4])
        h2 = tanh(p[5]*x1 + p[6]*x2 + p[7]*t + p[8])
        
        dx[1] = p[13] * h1  # dx1/dt
        dx[2] = p[14] * h2  # dx2/dt
    end
    
    # Make BNN-ODE predictions
    println("  → Computing BNN-ODE predictions...")
    bnn_predictions = []
    
    for i in 1:length(t_derivatives)
        x = Y_test[i, :]
        t = t_derivatives[i]
        dx = zeros(2)
        bnn_dynamics!(dx, x, bnn_params, t)
        push!(bnn_predictions, dx)
    end
    
    bnn_predictions = hcat(bnn_predictions...)'
    println("  → BNN-ODE predictions range: x1 ($(minimum(bnn_predictions[:, 1])), $(maximum(bnn_predictions[:, 1]))), x2 ($(minimum(bnn_predictions[:, 2])), $(maximum(bnn_predictions[:, 2])))")
    
    # Calculate BNN-ODE metrics
    bnn_errors = bnn_predictions .- actual_derivatives
    
    # MSE
    bnn_mse_x1 = mean(bnn_errors[:, 1].^2)
    bnn_mse_x2 = mean(bnn_errors[:, 2].^2)
    bnn_mse_total = mean(bnn_errors.^2)
    
    # RMSE
    bnn_rmse_x1 = sqrt(bnn_mse_x1)
    bnn_rmse_x2 = sqrt(bnn_mse_x2)
    bnn_rmse_total = sqrt(bnn_mse_total)
    
    # MAE
    bnn_mae_x1 = mean(abs.(bnn_errors[:, 1]))
    bnn_mae_x2 = mean(abs.(bnn_errors[:, 2]))
    bnn_mae_total = mean(abs.(bnn_errors))
    
    # R²
    bnn_r2_x1 = 1 - sum(bnn_errors[:, 1].^2) / sum((actual_derivatives[:, 1] .- mean(actual_derivatives[:, 1])).^2)
    bnn_r2_x2 = 1 - sum(bnn_errors[:, 2].^2) / sum((actual_derivatives[:, 2] .- mean(actual_derivatives[:, 2])).^2)
    bnn_r2_total = 1 - sum(bnn_errors.^2) / sum((actual_derivatives .- mean(actual_derivatives, dims=1)).^2)
    
    results["BNN-ODE"] = Dict(
        "MSE_x1" => bnn_mse_x1,
        "MSE_x2" => bnn_mse_x2,
        "MSE_total" => bnn_mse_total,
        "RMSE_x1" => bnn_rmse_x1,
        "RMSE_x2" => bnn_rmse_x2,
        "RMSE_total" => bnn_rmse_total,
        "MAE_x1" => bnn_mae_x1,
        "MAE_x2" => bnn_mae_x2,
        "MAE_total" => bnn_mae_total,
        "R2_x1" => bnn_r2_x1,
        "R2_x2" => bnn_r2_x2,
        "R2_total" => bnn_r2_total
    )
    
    println("✅ BNN-ODE evaluation completed")
else
    println("❌ BNN-ODE model not found at $bnn_path")
end

# ============================================================================
# UDE EVALUATION
# ============================================================================

println("\n" * "="^60)
println("UDE EVALUATION")
println("="^60)

# Load UDE model
ude_path = "checkpoints/ude_results_fixed.bson"
if isfile(ude_path)
    println("✅ Loading UDE model...")
    ude_data = BSON.load(ude_path)
    ude_results = ude_data[:ude_results]
    
    println("  → Physics parameters: $(length(ude_results[:physics_params_mean]))")
    println("  → Neural parameters: $(length(ude_results[:neural_params_mean]))")
    
    # Get UDE parameters
    physics_params = ude_results[:physics_params_mean]
    neural_params = ude_results[:neural_params_mean]
    println("  → Physics param range: ($(minimum(physics_params)), $(maximum(physics_params)))")
    println("  → Neural param range: ($(minimum(neural_params)), $(maximum(neural_params)))")
    
    # Define UDE dynamics function
    function ude_dynamics!(dx, x, p, t)
        x1, x2 = x
        ηin, ηout, α, β, γ = p[1:5]  # Physics parameters
        nn_params = p[6:end]         # Neural parameters
        
        # Control input and generation/load
        u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
        Pgen = max(0, sin((t - 6) * π / 12))
        Pload = 0.6 + 0.2 * sin(t * π / 12)
        
        # Physics-based terms
        Pin = u > 0 ? ηin * u : (1 / ηout) * u
        dx[1] = Pin - Pload  # Energy balance
        
        # Neural network for nonlinear term
        h1 = tanh(nn_params[1]*x1 + nn_params[2]*x2 + nn_params[3]*Pgen + nn_params[4]*Pload + nn_params[5]*t + nn_params[6])
        h2 = tanh(nn_params[7]*x1 + nn_params[8]*x2 + nn_params[9]*Pgen + nn_params[10]*Pload + nn_params[11]*t + nn_params[12])
        nn_output = nn_params[13]*h1 + nn_params[14]*h2 + nn_params[15]
        
        dx[2] = -α * x2 + nn_output + γ * x1  # Power balance with neural term
    end
    
    # Make UDE predictions
    println("  → Computing UDE predictions...")
    ude_predictions = []
    p_ude = [physics_params..., neural_params...]
    
    for i in 1:length(t_derivatives)
        x = Y_test[i, :]
        t = t_derivatives[i]
        dx = zeros(2)
        ude_dynamics!(dx, x, p_ude, t)
        push!(ude_predictions, dx)
    end
    
    ude_predictions = hcat(ude_predictions...)'
    println("  → UDE predictions range: x1 ($(minimum(ude_predictions[:, 1])), $(maximum(ude_predictions[:, 1]))), x2 ($(minimum(ude_predictions[:, 2])), $(maximum(ude_predictions[:, 2])))")
    
    # Calculate UDE metrics
    ude_errors = ude_predictions .- actual_derivatives
    
    # MSE
    ude_mse_x1 = mean(ude_errors[:, 1].^2)
    ude_mse_x2 = mean(ude_errors[:, 2].^2)
    ude_mse_total = mean(ude_errors.^2)
    
    # RMSE
    ude_rmse_x1 = sqrt(ude_mse_x1)
    ude_rmse_x2 = sqrt(ude_mse_x2)
    ude_rmse_total = sqrt(ude_mse_total)
    
    # MAE
    ude_mae_x1 = mean(abs.(ude_errors[:, 1]))
    ude_mae_x2 = mean(abs.(ude_errors[:, 2]))
    ude_mae_total = mean(abs.(ude_errors))
    
    # R²
    ude_r2_x1 = 1 - sum(ude_errors[:, 1].^2) / sum((actual_derivatives[:, 1] .- mean(actual_derivatives[:, 1])).^2)
    ude_r2_x2 = 1 - sum(ude_errors[:, 2].^2) / sum((actual_derivatives[:, 2] .- mean(actual_derivatives[:, 2])).^2)
    ude_r2_total = 1 - sum(ude_errors.^2) / sum((actual_derivatives .- mean(actual_derivatives, dims=1)).^2)
    
    results["UDE"] = Dict(
        "MSE_x1" => ude_mse_x1,
        "MSE_x2" => ude_mse_x2,
        "MSE_total" => ude_mse_total,
        "RMSE_x1" => ude_rmse_x1,
        "RMSE_x2" => ude_rmse_x2,
        "RMSE_total" => ude_rmse_total,
        "MAE_x1" => ude_mae_x1,
        "MAE_x2" => ude_mae_x2,
        "MAE_total" => ude_mae_total,
        "R2_x1" => ude_r2_x1,
        "R2_x2" => ude_r2_x2,
        "R2_total" => ude_r2_total
    )
    
    println("✅ UDE evaluation completed")
else
    println("❌ UDE model not found at $ude_path")
end

# ============================================================================
# BASELINE COMPARISON
# ============================================================================

println("\n" * "="^60)
println("BASELINE COMPARISON")
println("="^60)

# Linear baseline (mean prediction)
linear_pred_x1 = fill(mean(actual_derivatives[:, 1]), size(actual_derivatives, 1))
linear_pred_x2 = fill(mean(actual_derivatives[:, 2]), size(actual_derivatives, 1))
linear_pred = hcat(linear_pred_x1, linear_pred_x2)

linear_errors = linear_pred .- actual_derivatives

# Calculate baseline metrics
linear_mse_x1 = mean(linear_errors[:, 1].^2)
linear_mse_x2 = mean(linear_errors[:, 2].^2)
linear_mse_total = mean(linear_errors.^2)

linear_rmse_x1 = sqrt(linear_mse_x1)
linear_rmse_x2 = sqrt(linear_mse_x2)
linear_rmse_total = sqrt(linear_mse_total)

linear_mae_x1 = mean(abs.(linear_errors[:, 1]))
linear_mae_x2 = mean(abs.(linear_errors[:, 2]))
linear_mae_total = mean(abs.(linear_errors))

linear_r2_x1 = 1 - sum(linear_errors[:, 1].^2) / sum((actual_derivatives[:, 1] .- mean(actual_derivatives[:, 1])).^2)
linear_r2_x2 = 1 - sum(linear_errors[:, 2].^2) / sum((actual_derivatives[:, 2] .- mean(actual_derivatives[:, 2])).^2)
linear_r2_total = 1 - sum(linear_errors.^2) / sum((actual_derivatives .- mean(actual_derivatives, dims=1)).^2)

results["Linear_Baseline"] = Dict(
    "MSE_x1" => linear_mse_x1,
    "MSE_x2" => linear_mse_x2,
    "MSE_total" => linear_mse_total,
    "RMSE_x1" => linear_rmse_x1,
    "RMSE_x2" => linear_rmse_x2,
    "RMSE_total" => linear_rmse_total,
    "MAE_x1" => linear_mae_x1,
    "MAE_x2" => linear_mae_x2,
    "MAE_total" => linear_mae_total,
    "R2_x1" => linear_r2_x1,
    "R2_x2" => linear_r2_x2,
    "R2_total" => linear_r2_total
)

println("✅ Baseline evaluation completed")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

println("\n" * "="^60)
println("COMPREHENSIVE RESULTS SUMMARY")
println("="^60)

# Print detailed results
for (model_name, metrics) in results
    println("\n📊 $model_name:")
    println("  MSE (x1, x2, total): ($(@sprintf("%.6f", metrics["MSE_x1"])), $(@sprintf("%.6f", metrics["MSE_x2"])), $(@sprintf("%.6f", metrics["MSE_total"])))")
    println("  RMSE (x1, x2, total): ($(@sprintf("%.6f", metrics["RMSE_x1"])), $(@sprintf("%.6f", metrics["RMSE_x2"])), $(@sprintf("%.6f", metrics["RMSE_total"])))")
    println("  MAE (x1, x2, total): ($(@sprintf("%.6f", metrics["MAE_x1"])), $(@sprintf("%.6f", metrics["MAE_x2"])), $(@sprintf("%.6f", metrics["MAE_total"])))")
    println("  R² (x1, x2, total): ($(@sprintf("%.6f", metrics["R2_x1"])), $(@sprintf("%.6f", metrics["R2_x2"])), $(@sprintf("%.6f", metrics["R2_total"])))")
end

# Save results to CSV
println("\n💾 Saving results...")
results_df = DataFrame()

# Define columns first
columns = ["Model", "MSE_x1", "MSE_x2", "MSE_total", "RMSE_x1", "RMSE_x2", "RMSE_total", "MAE_x1", "MAE_x2", "MAE_total", "R2_x1", "R2_x2", "R2_total"]
for col in columns
    results_df[!, col] = []
end

for (model_name, metrics) in results
    row = [model_name, metrics["MSE_x1"], metrics["MSE_x2"], metrics["MSE_total"], 
           metrics["RMSE_x1"], metrics["RMSE_x2"], metrics["RMSE_total"],
           metrics["MAE_x1"], metrics["MAE_x2"], metrics["MAE_total"],
           metrics["R2_x1"], metrics["R2_x2"], metrics["R2_total"]]
    push!(results_df, row)
end

CSV.write("results/comprehensive_model_comparison.csv", results_df)
println("  → Results saved to results/comprehensive_model_comparison.csv")

println("\n✅ Comprehensive evaluation completed!") 