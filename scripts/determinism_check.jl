#!/usr/bin/env julia

"""
Determinism validation script for reproducibility

Runs training with multiple seeds and validates that results are deterministic
within reasonable bounds. Saves a summary table with mean±SD for key metrics.

Usage: julia scripts/determinism_check.jl [--seeds=5] [--modeltype=bnn|ude|both]
"""

using Random, Statistics, CSV, DataFrames, BSON, Dates
using Printf, LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "training.jl"))
using .Training

function parse_args(argv)
    opts = Dict{String,Any}("seeds" => 5, "modeltype" => "both")
    for a in argv
        if startswith(a, "--seeds=")
            opts["seeds"] = parse(Int, split(a, "=", limit=2)[2])
        elseif startswith(a, "--modeltype=")
            opts["modeltype"] = split(a, "=", limit=2)[2]
        end
    end
    return opts
end

function run_determinism_check()
    opts = parse_args(ARGS)
    n_seeds = opts["seeds"]
    modeltype = opts["modeltype"]
    
    println("🧪 Determinism Check Starting")
    println("  → Testing $n_seeds seeds for modeltype: $modeltype")
    println("  → Timestamp: $(Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
    
    results = Dict{String, Vector{Any}}()
    
    # Test BNN if requested
    if modeltype in ["bnn", "both"]
        println("\n🔬 Testing Bayesian Neural ODE determinism...")
        bnn_results = []
        
        for seed in 42:(41+n_seeds)
            println("  → Seed $seed...")
            cfg = Dict("train" => Dict("seed" => seed, "samples" => 100, "warmup" => 50))  # Small for speed
            
            res = Training.train!(modeltype=:bnn, cfg=cfg)
            push!(bnn_results, Dict(
                :seed => seed,
                :params_mean_norm => norm(res[:params_mean]),
                :noise_mean => res[:noise_mean],
                :n_samples => res[:n_samples],
                :metadata => res[:metadata]
            ))
        end
        results["bnn"] = bnn_results
    end
    
    # Test UDE if requested  
    if modeltype in ["ude", "both"]
        println("\n🔬 Testing UDE determinism...")
        ude_results = []
        
        for seed in 42:(41+n_seeds)
            println("  → Seed $seed...")
            cfg = Dict("train" => Dict("seed" => seed, "samples" => 100, "warmup" => 50))  # Small for speed
            
            res = Training.train!(modeltype=:ude, cfg=cfg)
            push!(ude_results, Dict(
                :seed => seed,
                :physics_mean_norm => norm(res[:physics_params_mean]),
                :neural_mean_norm => norm(res[:neural_params_mean]),
                :noise_mean => res[:noise_mean],
                :n_samples => res[:n_samples],
                :metadata => res[:metadata]
            ))
        end
        results["ude"] = ude_results
    end
    
    # Compute summary statistics
    summary_table = []
    
    if haskey(results, "bnn")
        bnn_res = results["bnn"]
        params_norms = [r[:params_mean_norm] for r in bnn_res]
        noise_means = [r[:noise_mean] for r in bnn_res]
        
        push!(summary_table, Dict(
            "model" => "BNN-ODE",
            "metric" => "params_norm",
            "mean" => mean(params_norms),
            "std" => std(params_norms),
            "cv" => std(params_norms) / mean(params_norms)
        ))
        push!(summary_table, Dict(
            "model" => "BNN-ODE", 
            "metric" => "noise_level",
            "mean" => mean(noise_means),
            "std" => std(noise_means),
            "cv" => std(noise_means) / mean(noise_means)
        ))
    end
    
    if haskey(results, "ude")
        ude_res = results["ude"]
        physics_norms = [r[:physics_mean_norm] for r in ude_res]
        neural_norms = [r[:neural_mean_norm] for r in ude_res] 
        noise_means = [r[:noise_mean] for r in ude_res]
        
        push!(summary_table, Dict(
            "model" => "UDE",
            "metric" => "physics_norm", 
            "mean" => mean(physics_norms),
            "std" => std(physics_norms),
            "cv" => std(physics_norms) / mean(physics_norms)
        ))
        push!(summary_table, Dict(
            "model" => "UDE",
            "metric" => "neural_norm",
            "mean" => mean(neural_norms),
            "std" => std(neural_norms), 
            "cv" => std(neural_norms) / mean(neural_norms)
        ))
        push!(summary_table, Dict(
            "model" => "UDE",
            "metric" => "noise_level",
            "mean" => mean(noise_means),
            "std" => std(noise_means),
            "cv" => std(noise_means) / mean(noise_means)
        ))
    end
    
    # Save results
    results_dir = joinpath(@__DIR__, "..", "paper", "results")
    mkpath(results_dir)
    
    df = DataFrame(summary_table)
    CSV.write(joinpath(results_dir, "determinism_check.csv"), df)
    
    # Print summary
    println("\n📊 Determinism Check Results:")
    println("=" ^ 60)
    for row in eachrow(df)
        @printf("%-10s %-15s: %.4f ± %.4f (CV: %.4f)\n", 
               row.model, row.metric, row.mean, row.std, row.cv)
    end
    println("=" ^ 60)
    
    # Check for issues
    high_cv = any(df.cv .> 0.1)  # Coefficient of variation > 10%
    if high_cv
        println("⚠️  WARNING: High variability detected (CV > 10%)")
        println("   This may indicate non-deterministic behavior or insufficient warmup")
    else
        println("✅ PASS: All metrics show acceptable variability (CV ≤ 10%)")
    end
    
    println("\n📁 Results saved to: paper/results/determinism_check.csv")
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_determinism_check()
end 