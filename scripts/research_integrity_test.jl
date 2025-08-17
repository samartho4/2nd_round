#!/usr/bin/env julia

using CSV
using DataFrames
using Statistics
using BSON
using Test
using Random

println("🔬 RESEARCH INTEGRITY TESTING")
println("=" ^ 50)

# Set random seed for reproducibility
Random.seed!(42)

# Test 1: Data Integrity
println("\n📊 TEST 1: DATA INTEGRITY")
println("-" ^ 30)

# Load datasets
println("Loading datasets...")
train_data = CSV.read("data/training_dataset.csv", DataFrame)
val_data = CSV.read("data/validation_dataset.csv", DataFrame)
test_data = CSV.read("data/test_dataset.csv", DataFrame)

# Check data dimensions
println("Training data: $(nrow(train_data)) rows, $(ncol(train_data)) columns")
println("Validation data: $(nrow(val_data)) rows, $(ncol(val_data)) columns")
println("Test data: $(nrow(test_data)) rows, $(ncol(test_data)) columns")

# Verify data structure
@test hasproperty(train_data, :time)
@test hasproperty(train_data, :x1)
@test hasproperty(train_data, :x2)
@test hasproperty(train_data, :scenario)

# Check for missing values
println("Missing values check:")
println("Training: $(sum(ismissing.(train_data)))")
println("Validation: $(sum(ismissing.(val_data)))")
println("Test: $(sum(ismissing.(test_data)))")

# Check data ranges
println("\nData ranges:")
println("x1 (SOC) range: $(minimum(train_data.x1)) to $(maximum(train_data.x1))")
println("x2 (Power) range: $(minimum(train_data.x2)) to $(maximum(train_data.x2))")

# Check scenario distribution
scenario_counts = combine(groupby(train_data, :scenario), nrow => :count)
println("\nScenario distribution:")
println("Unique scenarios: $(nrow(scenario_counts))")
println("Min samples per scenario: $(minimum(scenario_counts.count))")
println("Max samples per scenario: $(maximum(scenario_counts.count))")

# Test 2: Model Loading and Basic Properties
println("\n🤖 TEST 2: MODEL INTEGRITY")
println("-" ^ 30)

# Load models
println("Loading models...")
try
    bnn_results = BSON.load("checkpoints/bayesian_neural_ode_results.bson")
    println("✅ BNN-ODE model loaded successfully")
    println("Keys: $(keys(bnn_results))")
catch e
    println("❌ Failed to load BNN-ODE model: $e")
end

try
    ude_results = BSON.load("checkpoints/ude_results_fixed.bson")
    println("✅ UDE model loaded successfully")
    println("Keys: $(keys(ude_results))")
catch e
    println("❌ Failed to load UDE model: $e")
end

# Test 3: Data Leakage Check
println("\n🔍 TEST 3: DATA LEAKAGE ANALYSIS")
println("-" ^ 30)

# Check for scenario overlap between train/val/test
train_scenarios = unique(train_data.scenario)
val_scenarios = unique(val_data.scenario)
test_scenarios = unique(test_data.scenario)

train_val_overlap = intersect(train_scenarios, val_scenarios)
train_test_overlap = intersect(train_scenarios, test_scenarios)
val_test_overlap = intersect(val_scenarios, test_scenarios)

println("Train-Val overlap: $(length(train_val_overlap)) scenarios")
println("Train-Test overlap: $(length(train_test_overlap)) scenarios")
println("Val-Test overlap: $(length(val_test_overlap)) scenarios")

if length(train_test_overlap) > 0
    println("⚠️  WARNING: Data leakage detected! Test scenarios appear in training data:")
    println(train_test_overlap)
end

# Test 4: Statistical Consistency
println("\n📈 TEST 4: STATISTICAL CONSISTENCY")
println("-" ^ 30)

# Check if performance claims are reasonable
println("Checking performance claims from summary...")

# Load performance results if available
if isfile("results/simple_model_comparison.csv")
    perf_data = CSV.read("results/simple_model_comparison.csv", DataFrame)
    println("Performance data loaded: $(nrow(perf_data)) rows")
    
    # Check for extreme performance differences
    if hasproperty(perf_data, :mse_x1)
        bnn_mse_x1 = perf_data[perf_data.model .== "BNN-ODE", :mse_x1][1]
        ude_mse_x1 = perf_data[perf_data.model .== "UDE", :mse_x1][1]
        
        println("BNN-ODE MSE x1: $bnn_mse_x1")
        println("UDE MSE x1: $ude_mse_x1")
        
        if bnn_mse_x1 < 1e-5
            println("⚠️  WARNING: Extremely low BNN-ODE MSE - possible overfitting or data leakage")
        end
        
        if ude_mse_x1 > 0.1
            println("⚠️  WARNING: High UDE MSE - possible underfitting or data issues")
        end
    end
end

# Test 5: Reproducibility Check
println("\n🔄 TEST 5: REPRODUCIBILITY")
println("-" ^ 30)

# Check if random seeds are set
println("Random seed check: $(Random.seed!())")

# Check for deterministic operations
println("Testing deterministic operations...")
test_values = randn(100)
test_mean = mean(test_values)
println("Test mean: $test_mean")

# Test 6: Model Complexity Analysis
println("\n🧮 TEST 6: MODEL COMPLEXITY")
println("-" ^ 30)

# Check model file sizes
bnn_size = filesize("checkpoints/bayesian_neural_ode_results.bson")
ude_size = filesize("checkpoints/ude_results_fixed.bson")

println("BNN-ODE model size: $(bnn_size) bytes")
println("UDE model size: $(ude_size) bytes")

# Estimate parameter counts from file sizes
println("Estimated parameters (rough):")
println("BNN-ODE: ~$(round(Int, bnn_size/100)) parameters")
println("UDE: ~$(round(Int, ude_size/100)) parameters")

# Test 7: Evaluation Methodology
println("\n📋 TEST 7: EVALUATION METHODOLOGY")
println("-" ^ 30)

# Check if evaluation scripts exist
eval_scripts = ["scripts/evaluate.jl", "scripts/simple_model_comparison.jl"]
for script in eval_scripts
    if isfile(script)
        println("✅ $script exists")
    else
        println("❌ $script missing")
    end
end

# Check for proper train/val/test split
total_samples = nrow(train_data) + nrow(val_data) + nrow(test_data)
train_ratio = nrow(train_data) / total_samples
val_ratio = nrow(val_data) / total_samples
test_ratio = nrow(test_data) / total_samples

println("\nData split ratios:")
println("Train: $(round(train_ratio*100, digits=1))%")
println("Validation: $(round(val_ratio*100, digits=1))%")
println("Test: $(round(test_ratio*100, digits=1))%")

# Test 8: Documentation Consistency
println("\n📚 TEST 8: DOCUMENTATION CONSISTENCY")
println("-" ^ 30)

# Check if documentation matches actual data
claimed_train_samples = 7334
actual_train_samples = nrow(train_data)

println("Claimed training samples: $claimed_train_samples")
println("Actual training samples: $actual_train_samples")

if claimed_train_samples != actual_train_samples
    println("❌ INCONSISTENCY: Training sample count mismatch!")
else
    println("✅ Training sample count consistent")
end

# Test 9: Critical Issues Summary
println("\n🚨 CRITICAL ISSUES SUMMARY")
println("-" ^ 30)

issues_found = 0

# Check for data leakage
if length(train_test_overlap) > 0
    println("❌ CRITICAL: Data leakage detected")
    issues_found += 1
end

# Check for missing values
if sum(ismissing.(train_data)) > 0
    println("❌ CRITICAL: Missing values in training data")
    issues_found += 1
end

# Check for extreme performance
if isfile("results/simple_model_comparison.csv")
    perf_data = CSV.read("results/simple_model_comparison.csv", DataFrame)
    if hasproperty(perf_data, :mse_x1)
        bnn_mse_x1 = perf_data[perf_data.model .== "BNN-ODE", :mse_x1][1]
        if bnn_mse_x1 < 1e-6
            println("❌ CRITICAL: Suspiciously low BNN-ODE MSE - possible overfitting")
            issues_found += 1
        end
    end
end

# Check model files
if !isfile("checkpoints/bayesian_neural_ode_results.bson")
    println("❌ CRITICAL: BNN-ODE model file missing")
    issues_found += 1
end

if !isfile("checkpoints/ude_results_fixed.bson")
    println("❌ CRITICAL: UDE model file missing")
    issues_found += 1
end

if issues_found == 0
    println("✅ No critical issues found")
else
    println("❌ $issues_found critical issues found")
end

println("\n" ^ 50)
println("🔬 RESEARCH INTEGRITY TEST COMPLETE")
println("=" ^ 50) 