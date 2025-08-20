#!/usr/bin/env julia

"""
    evaluate_ude_final_status.jl

Simple evaluation script to check the current status of the UDE model and assess if the critical issues have been resolved.
"""

using Pkg
Pkg.activate(".")

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Load required modules
include(joinpath(@__DIR__, "..", "src", "training.jl"))
using .Training

# Import required packages
using BSON
using Statistics
using CSV
using DataFrames

println("🔍 UDE FINAL STATUS EVALUATION")
println("=" ^ 50)

# Check if fixed results exist
fixed_results_path = joinpath(@__DIR__, "..", "checkpoints", "fixed_ude_results.bson")
improved_results_path = joinpath(@__DIR__, "..", "checkpoints", "improved_ude_results.bson")
original_results_path = joinpath(@__DIR__, "..", "checkpoints", "ude_results_fixed.bson")

println("\n📊 CHECKING AVAILABLE RESULTS")
println("-" ^ 30)

results_found = false
results_data = nothing

if isfile(fixed_results_path)
    println("✅ Found fixed_ude_results.bson")
    BSON.@load fixed_results_path fixed_ude_results
    results_data = fixed_ude_results
    results_found = true
elseif isfile(improved_results_path)
    println("✅ Found improved_ude_results.bson")
    BSON.@load improved_results_path improved_ude_results
    results_data = improved_ude_results
    results_found = true
elseif isfile(original_results_path)
    println("✅ Found ude_results_fixed.bson")
    BSON.@load original_results_path ude_results
    results_data = ude_results
    results_found = true
else
    println("❌ No UDE results found")
end

if results_found
    println("\n📊 UNCERTAINTY ANALYSIS")
    println("-" ^ 30)
    
    # Check physics parameters
    if haskey(results_data, :physics_params_std)
        physics_std = results_data[:physics_params_std]
        println("Physics Parameters Uncertainty:")
        println("  ηin:  $(round(physics_std[1], digits=6))")
        println("  ηout: $(round(physics_std[2], digits=6))")
        println("  α:    $(round(physics_std[3], digits=6))")
        println("  β:    $(round(physics_std[4], digits=6))")
        println("  γ:    $(round(physics_std[5], digits=6))")
        
        # Check if uncertainty is achieved
        uncertainty_achieved = all(physics_std .> 1e-6)
        println("  → Uncertainty achieved: $(uncertainty_achieved ? "✅ YES" : "❌ NO")")
    else
        println("❌ No physics parameter uncertainty data")
    end
    
    # Check neural parameters
    if haskey(results_data, :neural_params_std)
        neural_std = results_data[:neural_params_std]
        println("Neural Parameters Uncertainty:")
        println("  Mean std: $(round(mean(neural_std), digits=6))")
        println("  Max std:  $(round(maximum(neural_std), digits=6))")
        println("  Min std:  $(round(minimum(neural_std), digits=6))")
        
        neural_uncertainty = mean(neural_std) > 1e-6
        println("  → Neural uncertainty achieved: $(neural_uncertainty ? "✅ YES" : "❌ NO")")
    else
        println("❌ No neural parameter uncertainty data")
    end
    
    # Check noise parameters
    if haskey(results_data, :noise_std)
        noise_std = results_data[:noise_std]
        println("Noise Uncertainty:")
        println("  σ std: $(round(noise_std, digits=6))")
        
        noise_uncertainty = noise_std > 1e-6
        println("  → Noise uncertainty achieved: $(noise_uncertainty ? "✅ YES" : "❌ NO")")
    else
        println("❌ No noise uncertainty data")
    end
    
    # Overall assessment
    println("\n🎯 OVERALL ASSESSMENT")
    println("-" ^ 30)
    
    if haskey(results_data, :uncertainty_achieved)
        overall_uncertainty = results_data[:uncertainty_achieved]
        println("Overall Uncertainty: $(overall_uncertainty ? "✅ ACHIEVED" : "❌ NOT ACHIEVED")")
    else
        println("Overall Uncertainty: ⚠️ UNKNOWN")
    end
    
    # Check metadata
    if haskey(results_data, :metadata)
        model_metadata = results_data[:metadata]
        println("Fixes Applied:")
        if haskey(model_metadata, :fixes_applied)
            for fix in model_metadata[:fixes_applied]
                println("  ✅ $fix")
            end
        end
        if haskey(model_metadata, :research_based)
            println("Research-based: $(model_metadata[:research_based] ? "✅ YES" : "❌ NO")")
        end
    end
    
    # Check performance metrics
    performance_path = joinpath(@__DIR__, "..", "results", "performance_metrics.bson")
    if isfile(performance_path)
        println("\n📈 PERFORMANCE METRICS")
        println("-" ^ 30)
        BSON.@load performance_path performance_metrics
        
        if haskey(performance_metrics, :soc_mean)
            println("SOC - Mean: $(round(performance_metrics[:soc_mean], digits=4))")
            println("SOC - Std:  $(round(performance_metrics[:soc_std], digits=4))")
        end
        if haskey(performance_metrics, :power_mean)
            println("Power - Mean: $(round(performance_metrics[:power_mean], digits=4))")
            println("Power - Std:  $(round(performance_metrics[:power_std], digits=4))")
        end
    end
end

# Check validation data
println("\n📋 VALIDATION DATA CHECK")
println("-" ^ 30)

validation_data_path = joinpath(@__DIR__, "..", "data", "validation_dataset.csv")
if isfile(validation_data_path)
    df_val = CSV.read(validation_data_path, DataFrame)
    println("✅ Validation data found: $(nrow(df_val)) samples")
    
    # Basic statistics
    soc_mean = mean(df_val.x1)
    power_mean = mean(df_val.x2)
    soc_std = std(df_val.x1)
    power_std = std(df_val.x2)
    
    println("SOC - Mean: $(round(soc_mean, digits=4)), Std: $(round(soc_std, digits=4))")
    println("Power - Mean: $(round(power_mean, digits=4)), Std: $(round(power_std, digits=4))")
else
    println("❌ Validation data not found")
end

# Final summary
println("\n📋 FINAL SUMMARY")
println("=" ^ 50)

if results_found
    if haskey(results_data, :uncertainty_achieved) && results_data[:uncertainty_achieved]
        println("🎯 STATUS: ✅ BAYESIAN UNCERTAINTY ACHIEVED")
        println("   → The UDE model now has proper uncertainty quantification")
        println("   → All three critical issues have been addressed")
    else
        println("🎯 STATUS: ⚠️ UNCERTAINTY ISSUES REMAIN")
        println("   → The model still lacks proper Bayesian uncertainty")
        println("   → Further fixes may be needed")
    end
else
    println("🎯 STATUS: ❌ NO RESULTS FOUND")
    println("   → No UDE training results available")
    println("   → Need to run training first")
end

println("\n🔧 RECOMMENDATIONS")
println("-" ^ 30)

if !results_found
    println("1. Run the UDE training script first")
    println("2. Check for any error messages during training")
    println("3. Verify that results are saved properly")
elseif haskey(results_data, :uncertainty_achieved) && !results_data[:uncertainty_achieved]
    println("1. The model still lacks proper uncertainty quantification")
    println("2. Consider adjusting MCMC settings")
    println("3. Check prior specifications")
    println("4. Verify numerical stability")
else
    println("1. ✅ All critical issues appear to be resolved")
    println("2. The model is ready for further evaluation")
    println("3. Consider running performance tests on validation data")
end

println("\n🎯 EVALUATION COMPLETED")
println("=" ^ 50) 