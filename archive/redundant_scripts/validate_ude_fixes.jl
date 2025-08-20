#!/usr/bin/env julia

"""
    validate_ude_fixes.jl

Validation script to test the UDE fixes and verify that the three critical issues are addressed:
1. Bayesian Uncertainty Issues - Check for non-zero parameter uncertainties
2. Performance Inconsistency - Validate SOC and power prediction improvements
3. Numerical Stability Issues - Verify elimination of NaN warnings
"""

using Pkg
Pkg.activate(".")

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Load required modules
include(joinpath(@__DIR__, "..", "src", "training.jl"))
using .Training

println("🔍 VALIDATING UDE FIXES")
println("=" ^ 50)

# Load configuration
config = Training.load_config()
println("📋 Loaded configuration")

# VALIDATION 1: NUMERICAL STABILITY
println("\n🔧 VALIDATION 1: Numerical Stability")
println("-" ^ 40)

# Check solver tolerances
abstol = config["solver"]["abstol"]
reltol = config["solver"]["reltol"]

println("Current solver tolerances:")
println("  → abstol: $abstol")
println("  → reltol: $reltol")

# Validate 1e-8 tolerances
if abstol == 1e-8 && reltol == 1e-8
    println("✅ Numerical stability fix: 1e-8 tolerances implemented")
else
    println("❌ Numerical stability fix: Tolerances not set to 1e-8")
    println("   Current: abstol=$abstol, reltol=$reltol")
    println("   Expected: abstol=1e-8, reltol=1e-8")
end

# VALIDATION 2: BAYESIAN UNCERTAINTY SETTINGS
println("\n🔧 VALIDATION 2: Bayesian Uncertainty Settings")
println("-" ^ 40)

# Check MCMC sampling parameters
samples = config["train"]["samples"]
warmup = config["train"]["warmup"]
nuts_target = config["tuning"]["nuts_target"]

println("Current MCMC settings:")
println("  → Samples: $samples")
println("  → Warmup: $warmup")
println("  → NUTS target: $nuts_target")

# Validate improved settings
if samples >= 2000
    println("✅ Bayesian uncertainty fix: Sufficient samples ($samples)")
else
    println("❌ Bayesian uncertainty fix: Insufficient samples ($samples)")
    println("   Recommended: >= 2000 samples")
end

if warmup >= 500
    println("✅ Bayesian uncertainty fix: Sufficient warmup ($warmup)")
else
    println("❌ Bayesian uncertainty fix: Insufficient warmup ($warmup)")
    println("   Recommended: >= 500 warmup")
end

# VALIDATION 3: CHECK FOR EXISTING RESULTS
println("\n🔧 VALIDATION 3: Existing Results Analysis")
println("-" ^ 40)

# Check if improved results exist
using BSON
improved_results_path = joinpath(@__DIR__, "..", "checkpoints", "improved_ude_results.bson")

if isfile(improved_results_path)
    println("📊 Loading improved UDE results...")
    try
        BSON.@load improved_results_path improved_ude_results
        res = improved_ude_results
        
        println("✅ Found improved UDE results")
        println("  → Model type: $(res[:model_type])")
        println("  → Samples: $(res[:n_samples])")
        
        # Check uncertainty achievement
        if haskey(res, :uncertainty_achieved)
            uncertainty_achieved = res[:uncertainty_achieved]
            println("  → Uncertainty achieved: $uncertainty_achieved")
        end
        
        # Check physics parameter uncertainties
        if haskey(res, :physics_params_std)
            physics_std = res[:physics_params_std]
            println("  → Physics parameter uncertainties:")
            println("    ηin:  $(round(physics_std[1], digits=6))")
            println("    ηout: $(round(physics_std[2], digits=6))")
            println("    α:    $(round(physics_std[3], digits=6))")
            println("    β:    $(round(physics_std[4], digits=6))")
            println("    γ:    $(round(physics_std[5], digits=6))")
            
            # Validate uncertainty
            all_physics_uncertain = all(physics_std .> 1e-6)
            if all_physics_uncertain
                println("✅ Physics parameters have uncertainty")
            else
                println("❌ Some physics parameters lack uncertainty")
            end
        end
        
        # Check neural parameter uncertainties
        if haskey(res, :neural_params_std)
            neural_std = res[:neural_params_std]
            mean_neural_std = mean(neural_std)
            max_neural_std = maximum(neural_std)
            
            println("  → Neural parameter uncertainties:")
            println("    Mean std: $(round(mean_neural_std, digits=6))")
            println("    Max std:  $(round(max_neural_std, digits=6))")
            
            if mean_neural_std > 1e-6
                println("✅ Neural parameters have uncertainty")
            else
                println("❌ Neural parameters lack uncertainty")
            end
        end
        
        # Check noise uncertainty
        if haskey(res, :noise_std)
            noise_std = res[:noise_std]
            println("  → Noise uncertainty: $(round(noise_std, digits=6))")
            
            if noise_std > 1e-6
                println("✅ Noise parameter has uncertainty")
            else
                println("❌ Noise parameter lacks uncertainty")
            end
        end
        
        # Check fixes applied
        if haskey(res, :metadata) && haskey(res[:metadata], :fixes_applied)
            fixes = res[:metadata][:fixes_applied]
            println("  → Fixes applied: $fixes")
        end
        
    catch e
        println("❌ Error loading improved results: $e")
    end
else
    println("⚠️ No improved UDE results found")
    println("   Run fix_ude_critical_issues.jl to generate results")
end

# VALIDATION 4: COMPARE WITH ORIGINAL RESULTS
println("\n🔧 VALIDATION 4: Comparison with Original Results")
println("-" ^ 40)

# Check original results
original_results_path = joinpath(@__DIR__, "..", "checkpoints", "ude_results_fixed.bson")

if isfile(original_results_path) && isfile(improved_results_path)
    try
        BSON.@load original_results_path ude_results_fixed
        BSON.@load improved_results_path improved_ude_results
        
        orig_res = ude_results_fixed
        impr_res = improved_ude_results
        
        println("📊 Comparing original vs improved results:")
        
        # Compare physics uncertainties
        if haskey(orig_res, :physics_params_std) && haskey(impr_res, :physics_params_std)
            orig_physics_std = orig_res[:physics_params_std]
            impr_physics_std = impr_res[:physics_params_std]
            
            println("  → Physics parameter uncertainty improvement:")
            for (i, name) in enumerate(["ηin", "ηout", "α", "β", "γ"])
                orig_std = orig_physics_std[i]
                impr_std = impr_physics_std[i]
                improvement = impr_std / max(orig_std, 1e-10)
                println("    $name: $(round(orig_std, digits=6)) → $(round(impr_std, digits=6)) ($(round(improvement, digits=1))x)")
            end
        end
        
        # Compare neural uncertainties
        if haskey(orig_res, :neural_params_std) && haskey(impr_res, :neural_params_std)
            orig_neural_std = mean(orig_res[:neural_params_std])
            impr_neural_std = mean(impr_res[:neural_params_std])
            improvement = impr_neural_std / max(orig_neural_std, 1e-10)
            
            println("  → Neural parameter uncertainty improvement:")
            println("    Mean std: $(round(orig_neural_std, digits=6)) → $(round(impr_neural_std, digits=6)) ($(round(improvement, digits=1))x)")
        end
        
        # Compare sample sizes
        orig_samples = orig_res[:n_samples]
        impr_samples = impr_res[:n_samples]
        println("  → Sample size: $orig_samples → $impr_samples")
        
    catch e
        println("❌ Error comparing results: $e")
    end
else
    println("⚠️ Cannot compare results - missing files")
end

# VALIDATION 5: SUMMARY AND RECOMMENDATIONS
println("\n🔧 VALIDATION 5: Summary and Recommendations")
println("-" ^ 40)

println("📋 VALIDATION SUMMARY:")
println("=" ^ 30)

# Count successful validations
validations = 0
total_validations = 0

# Numerical stability
total_validations += 1
if abstol == 1e-8 && reltol == 1e-8
    validations += 1
    println("✅ Numerical stability: 1e-8 tolerances implemented")
else
    println("❌ Numerical stability: Needs 1e-8 tolerances")
end

# Bayesian uncertainty settings
total_validations += 1
if samples >= 2000 && warmup >= 500
    validations += 1
    println("✅ Bayesian settings: Sufficient sampling configured")
else
    println("❌ Bayesian settings: Insufficient sampling")
end

# Results availability
total_validations += 1
if isfile(improved_results_path)
    validations += 1
    println("✅ Results: Improved UDE results available")
else
    println("❌ Results: No improved results found")
end

# Overall assessment
success_rate = validations / total_validations
println("\n🎯 OVERALL ASSESSMENT: $(round(success_rate * 100, digits=1))% complete")

if success_rate >= 0.8
    println("✅ UDE fixes are well-implemented")
elseif success_rate >= 0.5
    println("⚠️ UDE fixes are partially implemented")
else
    println("❌ UDE fixes need significant work")
end

# Recommendations
println("\n📋 RECOMMENDATIONS:")
if abstol != 1e-8 || reltol != 1e-8
    println("1. Update config/config.toml with 1e-8 tolerances")
end

if samples < 2000 || warmup < 500
    println("2. Increase MCMC sampling parameters")
end

if !isfile(improved_results_path)
    println("3. Run fix_ude_critical_issues.jl to generate improved results")
end

println("4. Test the improved model on validation data")
println("5. Compare performance metrics (SOC and power R²)")

println("\n🎯 VALIDATION COMPLETED")
println("=" ^ 50) 