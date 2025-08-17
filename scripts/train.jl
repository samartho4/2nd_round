#!/usr/bin/env julia

"""
    train.jl

Simple training script that uses the Training module to train models with expanded data.
"""

using Pkg
Pkg.activate(".")

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Load the training module
include(joinpath(@__DIR__, "..", "src", "training.jl"))
using .Training

println("🚀 TRAINING WITH EXPANDED DATA")
println("=" ^ 50)

# Load configuration
config = Training.load_config()
println("📋 Loaded configuration")

# Train Bayesian Neural ODE model
println("\n🔬 Training Bayesian Neural ODE model...")
try
    bnn_results = Training.train!(modeltype=:bnn, cfg=config)
    println("✅ Bayesian Neural ODE training completed")
    println("  → Parameters: $(length(bnn_results[:params_mean]))")
    println("  → Noise std: $(round(bnn_results[:noise_std], digits=4))")
    println("  → Samples: $(bnn_results[:n_samples])")
catch e
    println("❌ Bayesian Neural ODE training failed: $e")
end

# Train UDE model
println("\n🔬 Training UDE model...")
try
    ude_results = Training.train!(modeltype=:ude, cfg=config)
    println("✅ UDE training completed")
    println("  → Parameters: $(length(ude_results[:params_mean]))")
    println("  → Noise std: $(round(ude_results[:noise_std], digits=4))")
    println("  → Samples: $(ude_results[:n_samples])")
catch e
    println("❌ UDE training failed: $e")
end

println("\n✅ Training pipeline completed!") 