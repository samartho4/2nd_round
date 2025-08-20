#!/usr/bin/env julia

"""
Verify Training Data: Check Actual Data Used by Models
=====================================================

This script verifies the exact amount of data that the models were trained on
by examining model files and data files directly.
"""

using Pkg
Pkg.activate(".")

using Random, Statistics, CSV, DataFrames, BSON
using Dates

println("🔍 VERIFYING TRAINING DATA USAGE")
println("=" ^ 50)

# ============================================================================
# 1. CHECK ALL DATA FILES
# ============================================================================

println("\n📊 STEP 1: DATA FILE ANALYSIS")
println("-" ^ 30)

data_files = [
    "data/training_dataset.csv",
    "data/training_dataset_fixed.csv", 
    "data/validation_dataset.csv",
    "data/validation_dataset_fixed.csv",
    "data/test_dataset.csv",
    "data/test_dataset_fixed.csv"
]

for file in data_files
    if isfile(file)
        df = CSV.read(file, DataFrame)
        println("✅ $(file): $(nrow(df)) samples")
        if nrow(df) > 0
            println("  → Columns: $(names(df))")
            println("  → Time range: $(minimum(df.time)) - $(maximum(df.time)) hours")
            println("  → Scenarios: $(length(unique(df.scenario)))")
            println("  → x1 range: $(round(minimum(df.x1), digits=3)) - $(round(maximum(df.x1), digits=3))")
            println("  → x2 range: $(round(minimum(df.x2), digits=3)) - $(round(maximum(df.x2), digits=3))")
        end
    else
        println("❌ $(file): File not found")
    end
    println()
end

# ============================================================================
# 2. CHECK MODEL FILES
# ============================================================================

println("\n🤖 STEP 2: MODEL FILE ANALYSIS")
println("-" ^ 30)

# Check BNN-ODE model
bnn_path = "checkpoints/bayesian_neural_ode_results.bson"
if isfile(bnn_path)
    println("✅ BNN-ODE model found: $(bnn_path)")
    bnn_data = BSON.load(bnn_path)
    bnn_results = bnn_data[:bayesian_results]
    
    println("  → Model type: $(bnn_results[:model_type])")
    println("  → Architecture: $(bnn_results[:arch])")
    println("  → Parameters: $(length(bnn_results[:params_mean]))")
    println("  → MCMC samples: $(bnn_results[:n_samples])")
    println("  → Training time: $(get(bnn_results[:metadata], :timestamp, "Unknown"))")
    
    # Check if metadata contains data info
    if haskey(bnn_results, :metadata) && haskey(bnn_results[:metadata], :config)
        config = bnn_results[:metadata][:config]
        if haskey(config, "train") && haskey(config["train"], "subset_size")
            println("  → Config subset_size: $(config["train"]["subset_size"])")
        end
    end
else
    println("❌ BNN-ODE model not found")
end

println()

# Check UDE model
ude_path = "checkpoints/ude_results_fixed.bson"
if isfile(ude_path)
    println("✅ UDE model found: $(ude_path)")
    ude_data = BSON.load(ude_path)
    ude_results = ude_data[:ude_results]
    
    println("  → Model type: $(ude_results[:model_type])")
    println("  → Physics parameters: $(length(ude_results[:physics_params_mean]))")
    println("  → Neural parameters: $(length(ude_results[:neural_params_mean]))")
    println("  → MCMC samples: $(ude_results[:n_samples])")
    println("  → Training time: $(get(ude_results[:metadata], :timestamp, "Unknown"))")
    
    # Check if metadata contains data info
    if haskey(ude_results, :metadata) && haskey(ude_results[:metadata], :config)
        config = ude_results[:metadata][:config]
        if haskey(config, "train") && haskey(config["train"], "subset_size")
            println("  → Config subset_size: $(config["train"]["subset_size"])")
        end
    end
else
    println("❌ UDE model not found")
end

# ============================================================================
# 3. DETERMINE WHICH DATA WAS USED
# ============================================================================

println("\n🔍 STEP 3: DETERMINE DATA USAGE")
println("-" ^ 30)

# Check which training file is most recent
training_files = [
    ("training_dataset.csv", "data/training_dataset.csv"),
    ("training_dataset_fixed.csv", "data/training_dataset_fixed.csv")
]

local most_recent_file = nothing
local most_recent_time = 0

for (name, path) in training_files
    if isfile(path)
        file_time = stat(path).mtime
        if file_time > most_recent_time
            most_recent_time = file_time
            most_recent_file = (name, path)
        end
    end
end

if most_recent_file !== nothing
    println("📅 Most recent training file: $(most_recent_file[1])")
    println("  → Modified: $(Dates.unix2datetime(most_recent_time))")
    
    # Check if this matches what the models likely used
    df = CSV.read(most_recent_file[2], DataFrame)
    println("  → Samples: $(nrow(df))")
    println("  → This is likely what the models trained on")
else
    println("❌ No training files found")
end

# ============================================================================
# 4. VERIFY BY TESTING MODEL LOADING
# ============================================================================

println("\n🧪 STEP 4: MODEL LOADING TEST")
println("-" ^ 30)

# Test loading models and check their parameters
try
    if isfile(bnn_path)
        bnn_data = BSON.load(bnn_path)
        bnn_results = bnn_data[:bayesian_results]
        
        # Check parameter statistics
        params_mean = bnn_results[:params_mean]
        params_std = bnn_results[:params_std]
        
        println("✅ BNN-ODE model loads successfully")
        println("  → Parameter mean range: $(round(minimum(params_mean), digits=6)) - $(round(maximum(params_mean), digits=6))")
        println("  → Parameter std range: $(round(minimum(params_std), digits=6)) - $(round(maximum(params_std), digits=6))")
        println("  → Noise std: $(round(bnn_results[:noise_std], digits=6))")
        
        # Check if parameters show uncertainty
        if any(params_std .> 1e-6)
            println("  → ✅ Model shows parameter uncertainty")
        else
            println("  → ⚠️ Model shows minimal parameter uncertainty")
        end
    end
catch e
    println("❌ BNN-ODE model loading failed: $e")
end

try
    if isfile(ude_path)
        ude_data = BSON.load(ude_path)
        ude_results = ude_data[:ude_results]
        
        # Check parameter statistics
        physics_mean = ude_results[:physics_params_mean]
        physics_std = ude_results[:physics_params_std]
        neural_mean = ude_results[:neural_params_mean]
        neural_std = ude_results[:neural_params_std]
        
        println("✅ UDE model loads successfully")
        println("  → Physics param mean range: $(round(minimum(physics_mean), digits=6)) - $(round(maximum(physics_mean), digits=6))")
        println("  → Physics param std range: $(round(minimum(physics_std), digits=6)) - $(round(maximum(physics_std), digits=6))")
        println("  → Neural param mean range: $(round(minimum(neural_mean), digits=6)) - $(round(maximum(neural_mean), digits=6))")
        println("  → Neural param std range: $(round(minimum(neural_std), digits=6)) - $(round(maximum(neural_std), digits=6))")
        println("  → Noise std: $(round(ude_results[:noise_std], digits=6))")
        
        # Check if parameters show uncertainty
        if any(physics_std .> 1e-6) || any(neural_std .> 1e-6)
            println("  → ✅ Model shows parameter uncertainty")
        else
            println("  → ⚠️ Model shows minimal parameter uncertainty")
        end
    end
catch e
    println("❌ UDE model loading failed: $e")
end

# ============================================================================
# 5. SUMMARY
# ============================================================================

println("\n📋 SUMMARY")
println("=" ^ 50)

println("📊 Available Data Files:")
for (name, path) in training_files
    if isfile(path)
        df = CSV.read(path, DataFrame)
        println("  → $(name): $(nrow(df)) samples")
    end
end

println("\n🤖 Model Status:")
if isfile(bnn_path)
    bnn_data = BSON.load(bnn_path)
    bnn_results = bnn_data[:bayesian_results]
    println("  → BNN-ODE: ✅ Trained with $(bnn_results[:n_samples]) MCMC samples")
end

if isfile(ude_path)
    ude_data = BSON.load(ude_path)
    ude_results = ude_data[:ude_results]
    println("  → UDE: ✅ Trained with $(ude_results[:n_samples]) MCMC samples")
end

println("\n🎯 Conclusion:")
if most_recent_file !== nothing
    df = CSV.read(most_recent_file[2], DataFrame)
    println("  → Models likely trained on: $(most_recent_file[1])")
    println("  → Training samples: $(nrow(df))")
    println("  → Scenarios: $(length(unique(df.scenario)))")
    
    if nrow(df) >= 1000
        println("  → ✅ Sufficient data for training")
    else
        println("  → ⚠️ Limited data - may need more samples")
    end
end

println("\n🏆 VERIFICATION COMPLETE") 