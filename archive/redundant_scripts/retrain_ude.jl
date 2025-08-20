#!/usr/bin/env julia

"""
Retrain UDE Model on Full Dataset
=================================

This script retrains the UDE model on the full 7,334 samples
to improve performance and uncertainty quantification.
"""

using Pkg
Pkg.activate(".")

using Random, Statistics, CSV, DataFrames, BSON
using Dates

# Add source directory to path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include training module
include(joinpath(@__DIR__, "..", "src", "training.jl"))
using .Training

println("🚀 RETRAINING UDE MODEL ON FULL DATASET")
println("=" ^ 50)

# Set seed for reproducibility
Random.seed!(42)

# Check current data
println("\n📊 DATA VERIFICATION")
println("-" ^ 30)

data_path = joinpath(@__DIR__, "..", "data", "training_dataset_fixed.csv")
if isfile(data_path)
    df = CSV.read(data_path, DataFrame)
    println("✅ Training data: $(nrow(df)) samples")
    println("  → Scenarios: $(length(unique(df.scenario)))")
    println("  → Time range: $(minimum(df.time)) - $(maximum(df.time)) hours")
    println("  → x1 range: $(round(minimum(df.x1), digits=3)) - $(round(maximum(df.x1), digits=3))")
    println("  → x2 range: $(round(minimum(df.x2), digits=3)) - $(round(maximum(df.x2), digits=3))")
else
    println("❌ Training data not found")
    exit(1)
end

# Load configuration
println("\n⚙️ CONFIGURATION")
println("-" ^ 30)

config = Training.load_config()
println("✅ Configuration loaded")
println("  → subset_size: $(get(config, "train", Dict()) |> d -> get(d, "subset_size", "Unknown"))")
println("  → samples: $(get(config, "train", Dict()) |> d -> get(d, "samples", "Unknown"))")
println("  → warmup: $(get(config, "train", Dict()) |> d -> get(d, "warmup", "Unknown"))")

# Retrain UDE model
println("\n🔄 RETRAINING UDE MODEL")
println("-" ^ 30)

start_time = time()

try
    println("Starting UDE training...")
    ude_results = Training.train!(modeltype=:ude, cfg=config)
    
    training_time = time() - start_time
    
    println("✅ UDE training completed successfully!")
    println("  → Training time: $(round(training_time, digits=1)) seconds")
    println("  → MCMC samples: $(ude_results[:n_samples])")
    println("  → Physics parameters: $(length(ude_results[:physics_params_mean]))")
    println("  → Neural parameters: $(length(ude_results[:neural_params_mean]))")
    println("  → Noise std: $(round(ude_results[:noise_std], digits=6))")
    
    # Check parameter uncertainty
    physics_std = ude_results[:physics_params_std]
    neural_std = ude_results[:neural_params_std]
    
    if any(physics_std .> 1e-6) || any(neural_std .> 1e-6)
        println("  → ✅ Model shows parameter uncertainty")
    else
        println("  → ⚠️ Model shows minimal parameter uncertainty")
    end
    
    # Save results summary
    results_summary = Dict(
        :training_time => training_time,
        :mcmc_samples => ude_results[:n_samples],
        :physics_params => length(ude_results[:physics_params_mean]),
        :neural_params => length(ude_results[:neural_params_mean]),
        :noise_std => ude_results[:noise_std],
        :has_uncertainty => any(physics_std .> 1e-6) || any(neural_std .> 1e-6),
        :timestamp => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    )
    
    BSON.@save joinpath(@__DIR__, "..", "results", "ude_retraining_summary.bson") results_summary
    
    println("\n💾 Results saved to:")
    println("  → checkpoints/ude_results_fixed.bson")
    println("  → results/ude_retraining_summary.bson")
    
catch e
    println("❌ UDE training failed: $e")
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

println("\n🏆 UDE RETRAINING COMPLETE") 