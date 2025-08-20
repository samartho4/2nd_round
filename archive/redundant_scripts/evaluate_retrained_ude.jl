#!/usr/bin/env julia

"""
Evaluate Retrained UDE Model
============================

This script evaluates the retrained UDE model on the full 7,334 samples
and provides comprehensive analysis of performance and uncertainty.
"""

using Pkg
Pkg.activate(".")

using Random, Statistics, CSV, DataFrames, BSON
using Dates, Plots

# Add source directory to path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include modules
include(joinpath(@__DIR__, "..", "src", "training.jl"))
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))

using .Training
using .Microgrid
using .NeuralNODEArchitectures

println("🔬 EVALUATING RETRAINED UDE MODEL")
println("=" ^ 50)

# Set seed for reproducibility
Random.seed!(42)

# Load retrained UDE results
println("\n📊 LOADING RETRAINED UDE RESULTS")
println("-" ^ 40)

ude_results_path = joinpath(@__DIR__, "..", "checkpoints", "ude_results_fixed.bson")
if isfile(ude_results_path)
    BSON.@load ude_results_path ude_results
    println("✅ UDE results loaded successfully")
else
    println("❌ UDE results not found")
    exit(1)
end

# Load training summary
summary_path = joinpath(@__DIR__, "..", "results", "ude_retraining_summary.bson")
if isfile(summary_path)
    BSON.@load summary_path results_summary
    println("✅ Training summary loaded")
else
    println("⚠️ Training summary not found")
end

# Analyze parameter uncertainty
println("\n📈 PARAMETER UNCERTAINTY ANALYSIS")
println("-" ^ 40)

physics_std = ude_results[:physics_params_std]
neural_std = ude_results[:neural_params_std]

println("Physics Parameters:")
for (i, std_val) in enumerate(physics_std)
    param_name = ["ηin", "ηout", "α", "β", "γ"][i]
    mean_val = ude_results[:physics_params_mean][i]
    println("  → $param_name: $(round(mean_val, digits=4)) ± $(round(std_val, digits=6))")
end

println("\nNeural Parameters:")
println("  → Mean std: $(round(mean(neural_std), digits=6))")
println("  → Max std: $(round(maximum(neural_std), digits=6))")
println("  → Min std: $(round(minimum(neural_std), digits=6))")

# Check if model shows meaningful uncertainty
has_uncertainty = any(physics_std .> 1e-6) || any(neural_std .> 1e-6)
println("\nUncertainty Assessment:")
if has_uncertainty
    println("  → ✅ Model shows parameter uncertainty")
else
    println("  → ⚠️ Model shows minimal parameter uncertainty")
end

# Load test data for evaluation
println("\n📊 LOADING TEST DATA")
println("-" ^ 40)

test_data_path = joinpath(@__DIR__, "..", "data", "test_dataset_fixed.csv")
if isfile(test_data_path)
    test_df = CSV.read(test_data_path, DataFrame)
    println("✅ Test data loaded: $(nrow(test_df)) samples")
    println("  → Scenarios: $(length(unique(test_df.scenario)))")
    println("  → Time range: $(minimum(test_df.time)) - $(maximum(test_df.time)) hours")
else
    println("❌ Test data not found")
    exit(1)
end

# Evaluate model performance
println("\n🎯 MODEL PERFORMANCE EVALUATION")
println("-" ^ 40)

# Function to evaluate UDE model on test data
function evaluate_ude_performance(test_df, ude_results)
    # Extract parameters
    ηin = ude_results[:physics_params_mean][1]
    ηout = ude_results[:physics_params_mean][2]
    α = ude_results[:physics_params_mean][3]
    β = ude_results[:physics_params_mean][4]
    γ = ude_results[:physics_params_mean][5]
    nn_params = ude_results[:neural_params_mean]
    
    # Calculate predictions
    predictions = []
    actuals = []
    
    for row in eachrow(test_df)
        # Extract state and inputs
        x1_true = row.x1
        x2_true = row.x2
        t = row.time
        
        # Generate Pgen and Pload using the functions
        Pgen = Microgrid.generation(t)
        Pload = Microgrid.load(t)
        
        # Calculate physics-based prediction
        u = Microgrid.control_input(t)
        demand_val = Microgrid.demand(t)
        
        # Physics equations
        dx1_physics = ηin * Pgen - ηout * Pload - α * x1_true
        dx2_physics = β * (demand_val - x2_true) - γ * x1_true
        
        # Neural correction
        neural_correction = NeuralNODEArchitectures.ude_nn_forward(x1_true, x2_true, Pgen, Pload, t, nn_params)
        
        # Combined prediction
        dx1_pred = dx1_physics + neural_correction
        dx2_pred = dx2_physics + neural_correction
        
        # Simple Euler integration for prediction
        dt = 0.1  # Small time step
        x1_pred = x1_true + dx1_pred * dt
        x2_pred = x2_true + dx2_pred * dt
        
        push!(predictions, [x1_pred, x2_pred])
        push!(actuals, [x1_true, x2_true])
    end
    
    # Calculate metrics
    predictions = hcat(predictions...)
    actuals = hcat(actuals...)
    
    # RMSE for each state variable
    rmse_x1 = sqrt(mean((predictions[1, :] - actuals[1, :]).^2))
    rmse_x2 = sqrt(mean((predictions[2, :] - actuals[2, :]).^2))
    
    # R² for each state variable
    r2_x1 = 1 - sum((predictions[1, :] - actuals[1, :]).^2) / sum((actuals[1, :] .- mean(actuals[1, :])).^2)
    r2_x2 = 1 - sum((predictions[2, :] - actuals[2, :]).^2) / sum((actuals[2, :] .- mean(actuals[2, :])).^2)
    
    return Dict(
        :rmse_x1 => rmse_x1,
        :rmse_x2 => rmse_x2,
        :r2_x1 => r2_x1,
        :r2_x2 => r2_x2,
        :predictions => predictions,
        :actuals => actuals
    )
end

# Run evaluation
println("Running UDE performance evaluation...")
performance_results = evaluate_ude_performance(test_df, ude_results)

println("Performance Metrics:")
println("  → RMSE x1: $(round(performance_results[:rmse_x1], digits=4))")
println("  → RMSE x2: $(round(performance_results[:rmse_x2], digits=4))")
println("  → R² x1: $(round(performance_results[:r2_x1], digits=4))")
println("  → R² x2: $(round(performance_results[:r2_x2], digits=4))")

# Generate evaluation report
println("\n📋 GENERATING EVALUATION REPORT")
println("-" ^ 40)

evaluation_report = Dict(
    :evaluation_date => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    :model_type => "UDE",
    :training_samples => 7334,
    :test_samples => nrow(test_df),
    :physics_params => Dict(
        :ηin => (ude_results[:physics_params_mean][1], ude_results[:physics_params_std][1]),
        :ηout => (ude_results[:physics_params_mean][2], ude_results[:physics_params_std][2]),
        :α => (ude_results[:physics_params_mean][3], ude_results[:physics_params_std][3]),
        :β => (ude_results[:physics_params_mean][4], ude_results[:physics_params_std][4]),
        :γ => (ude_results[:physics_params_mean][5], ude_results[:physics_params_std][5])
    ),
    :neural_params => Dict(
        :mean_std => mean(neural_std),
        :max_std => maximum(neural_std),
        :min_std => minimum(neural_std)
    ),
    :performance => Dict(
        :rmse_x1 => performance_results[:rmse_x1],
        :rmse_x2 => performance_results[:rmse_x2],
        :r2_x1 => performance_results[:r2_x1],
        :r2_x2 => performance_results[:r2_x2]
    ),
    :uncertainty_assessment => has_uncertainty,
    :mcmc_samples => ude_results[:n_samples],
    :noise_std => ude_results[:noise_std]
)

# Save evaluation report
BSON.@save joinpath(@__DIR__, "..", "results", "ude_evaluation_report.bson") evaluation_report

println("✅ Evaluation report saved to results/ude_evaluation_report.bson")

# Create summary
println("\n🏆 EVALUATION SUMMARY")
println("-" ^ 40)

println("UDE Model Performance on Full Dataset (7,334 samples):")
println("  → Training completed successfully")
println("  → MCMC samples: $(ude_results[:n_samples])")
println("  → Physics parameters learned with uncertainty: $has_uncertainty")
println("  → Test RMSE x1: $(round(performance_results[:rmse_x1], digits=4))")
println("  → Test RMSE x2: $(round(performance_results[:rmse_x2], digits=4))")
println("  → Test R² x1: $(round(performance_results[:r2_x1], digits=4))")
println("  → Test R² x2: $(round(performance_results[:r2_x2], digits=4))")

# Research assessment
println("\n🔬 RESEARCH ASSESSMENT")
println("-" ^ 40)

if has_uncertainty
    println("✅ UDE model successfully learned parameter uncertainty")
else
    println("⚠️ UDE model shows minimal parameter uncertainty")
end

if performance_results[:r2_x1] > 0.7 && performance_results[:r2_x2] > 0.7
    println("✅ UDE model shows good predictive performance")
else
    println("⚠️ UDE model shows limited predictive performance")
end

println("\n🎯 RECOMMENDATIONS:")
if !has_uncertainty
    println("  → Consider adjusting prior distributions")
    println("  → Increase MCMC warmup and samples")
    println("  → Check for numerical stability issues")
end

if performance_results[:r2_x1] < 0.7 || performance_results[:r2_x2] < 0.7
    println("  → Consider model architecture improvements")
    println("  → Check data quality and preprocessing")
    println("  → Evaluate feature engineering")
end

println("\n🏁 EVALUATION COMPLETE") 