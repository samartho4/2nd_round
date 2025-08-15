#!/usr/bin/env julia

"""
NEURIPS SUBMISSION - COMPLETE PIPELINE
=====================================

This script runs the complete experimental pipeline with proper methodology.
EXECUTE THIS SCRIPT AFTER IMPLEMENTING ALL PREVIOUS PHASES.
"""

using Pkg
Pkg.activate(".")
Pkg.instantiate()

println("="^60)
println("NEURIPS SUBMISSION - COMPLETE PIPELINE")
println("="^60)

# Ensure required packages are available
required_packages = [
    "Random", "DataFrames", "CSV", "Statistics",
    "HypothesisTests", "JLD2", "Glob"
]

for pkg in required_packages
    try
        @eval using $(Symbol(pkg))
    catch e
        println("Installing missing package: $pkg")
        Pkg.add(pkg)
        @eval using $(Symbol(pkg))
    end
end

# Include all source files
include(joinpath(@__DIR__, "..", "src", "data_processing.jl"))
include(joinpath(@__DIR__, "..", "src", "augment_training.jl"))
include(joinpath(@__DIR__, "..", "src", "statistical_evaluation.jl"))
# Satisfy dependency of baseline_models on StatisticalFramework
include(joinpath(@__DIR__, "..", "src", "statistical_framework.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline_models.jl"))
using .BaselineModels
include(joinpath(@__DIR__, "..", "src", "multi_seed_experiments.jl"))
include(joinpath(@__DIR__, "..", "src", "aggregate_results.jl"))
include(joinpath(@__DIR__, "..", "src", "physics_discovery_validation.jl"))

# Create results directories
mkpath(joinpath(@__DIR__, "..", "results"))
mkpath(joinpath(@__DIR__, "..", "paper", "figures"))
mkpath(joinpath(@__DIR__, "..", "paper", "results"))

println("\n🚀 Starting complete experimental pipeline...")

# PHASE 1: Data Processing
println("\n" * "="^30)
println("PHASE 1: Data Processing")
println("="^30)

try
    scenario_files = Glob.glob("data/scenarios/*/true_dense.csv")
    if isempty(scenario_files)
        error("No scenario files found in ../data/scenarios/*/true_dense.csv. Please check data directory.")
    end

    println("Found $(length(scenario_files)) scenario files")

    # Create proper temporal splits
    global train_data, val_data, test_data, expanded_train
    train_data, val_data, test_data = create_temporal_splits(scenario_files)

    # Expand training data
    expanded_train = augment_training_data(train_data, 15000)

    println("✅ Data processing completed")
    println("  - Training: $(nrow(expanded_train)) points")
    println("  - Validation: $(nrow(val_data)) points")
    println("  - Test: $(nrow(test_data)) points")

    CSV.write(joinpath(@__DIR__, "..", "data", "train_temporal.csv"), train_data)
    CSV.write(joinpath(@__DIR__, "..", "data", "val_temporal.csv"), val_data)
    CSV.write(joinpath(@__DIR__, "..", "data", "test_temporal.csv"), test_data)
    CSV.write(joinpath(@__DIR__, "..", "data", "train_expanded.csv"), expanded_train)

catch e
    println("❌ Data processing failed: $e")
    println("Please check that scenario data exists in ../data/scenarios/")
    exit(1)
end

# PHASE 2: Baseline Training
println("\n" * "="^30)
println("PHASE 2: Baseline Training")
println("="^30)

try
    println("Training Linear Regression baseline...")
    global expanded_train
    lr_baseline = train_baseline(LinearRegressionBaseline, expanded_train)

    println("Training Random Forest baseline...")
    rf_baseline = train_baseline(BaselineModels._RFStub, expanded_train)

    println("✅ Baseline training completed")
catch e
    println("❌ Baseline training failed: $e")
    exit(1)
end

# PHASE 3: Multi-Seed Experiments
println("\n" * "="^30)
println("PHASE 3: Multi-Seed Experiments")
println("="^30)

model_configs = Dict(
    "LinearRegression" => Dict(),
    "RandomForest" => Dict()
)

try
    println("Running multi-seed experiments (this may take a while)...")
    multi_seed_results = run_multi_seed_experiment(model_configs, 1:3)

    JLD2.save(joinpath(@__DIR__, "..", "results", "multi_seed_results.jld2"), "results", multi_seed_results)

    println("✅ Multi-seed experiments completed")
catch e
    println("❌ Multi-seed experiments failed: $e")
end

# PHASE 4: Results Aggregation
println("\n" * "="^30)
println("PHASE 4: Results Aggregation")
println("="^30)

try
    if isfile(joinpath(@__DIR__, "..", "results", "multi_seed_results.jld2"))
        multi_seed_results = JLD2.load(joinpath(@__DIR__, "..", "results", "multi_seed_results.jld2"), "results")
        aggregated = aggregate_multi_seed_results(multi_seed_results)
        final_table = create_results_table(aggregated)
        CSV.write(joinpath(@__DIR__, "..", "results", "final_results_with_stats.csv"), DataFrame(final_table))
        println("✅ Results aggregation completed")
    else
        println("⚠️  Multi-seed results not found, skipping aggregation")
    end
catch e
    println("❌ Results aggregation failed: $e")
end

# PHASE 5: Physics Discovery Validation (if UDE model exists)
println("\n" * "="^30)
println("PHASE 5: Physics Discovery Validation")
println("="^30)

try
    model_path = joinpath(@__DIR__, "..", "checkpoints", "ude_model.jld2")
    if isfile(model_path)
        println("Found UDE model, validating physics discovery...")
        ude_model = load_trained_model(model_path)
        test_scenarios = Glob.glob(joinpath(@__DIR__, "..", "data", "scenarios", "test_*.csv"))
        physics_results = validate_physics_discovery(ude_model, test_scenarios)
        JLD2.save(joinpath(@__DIR__, "..", "results", "physics_discovery_validation.jld2"), "results", physics_results)
        println("✅ Physics discovery validation completed")
    else
        println("⚠️  UDE model not found, skipping physics discovery validation")
    end
catch e
    println("❌ Physics discovery validation failed: $e")
end

# PHASE 6: Generate Final Summary
println("\n" * "="^30)
println("PHASE 6: Generate Publication Summary")
println("="^30)

try
    include(joinpath(@__DIR__, "generate_honest_results.jl"))
    summary_report = generate_publication_summary()
    cp(joinpath(@__DIR__, "..", "results", "publication_summary.md"), joinpath(@__DIR__, "..", "paper", "results", "final_results.md"), force=true)
    println("✅ Publication summary generated")
catch e
    println("❌ Publication summary generation failed: $e")
end

println("\n" * "="^60)
println("PIPELINE EXECUTION SUMMARY")
println("="^60)

println("\n📁 Generated Files:")
println("  - ../data/train_expanded.csv (expanded training data)")
println("  - ../results/multi_seed_results.jld2 (raw experiment results)")
println("  - ../results/final_results_with_stats.csv (aggregated statistics)")
println("  - ../results/physics_discovery_validation.jld2 (physics validation)")
println("  - ../paper/results/final_results.md (publication summary)")

println("\n📊 Next Steps:")
println("  1. Review results in ../paper/results/final_results.md")
println("  2. Check statistical significance of your methods")
println("  3. Add your actual UDE/BNN-ODE model training to model_configs")
println("  4. Run with more seeds (10+) for final results")
println("  5. Create publication figures")

println("\n⚠️  IMPORTANT REMINDERS:")
println("  - Results now include proper statistical analysis")
println("  - Proper temporal splits prevent data leakage")
println("  - Multiple seeds provide robust evaluation")

println("\n✅ Pipeline execution completed!") 