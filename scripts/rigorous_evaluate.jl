# scripts/rigorous_evaluate.jl
using Pkg
Pkg.activate(".")

using Random, Statistics
using JLD2, FileIO
include("../src/statistical_analysis.jl")
include("../src/baselines.jl")
include("../src/microgrid_system.jl")
include("../src/neural_ode_architectures.jl")

using .StatisticalAnalysis, .Baselines

function load_test_data()
    """
    Load test dataset - modify path as needed
    """
    # Replace with your actual data loading logic
    test_data = load("data/test_dataset.csv")  # or whatever format you use
    return test_data
end

function evaluate_single_run(model, test_data, seed::Int)
    """
    Evaluate model performance for a single random seed
    """
    Random.seed!(seed)
    
    # Your evaluation logic here
    # This should return a single performance metric (e.g., MSE)
    
    # Placeholder - replace with your actual evaluation
    predictions = predict_model(model, test_data)
    mse = mean((predictions .- test_data.targets).^2)
    
    return mse
end

function run_rigorous_evaluation()
    println("Starting rigorous evaluation with multiple seeds...")
    
    # Load test data
    test_data = load_test_data()
    
    # Define number of evaluation runs
    N_RUNS = 10
    SEEDS = 1:N_RUNS
    
    # Initialize results storage
    results = Dict{String, Vector{Float64}}()
    
    # Evaluate each model type
    model_configs = [
        "linear_baseline",
        "physics_only", 
        "bnn_ode",
        "ude"
    ]
    
    for model_name in model_configs
        println("Evaluating $model_name...")
        model_scores = Float64[]
        
        for seed in SEEDS
            println("  Run $seed/$N_RUNS")
            
            # Load or create model based on type
            if model_name == "linear_baseline"
                # Fit linear baseline
                model = fit_linear_baseline(test_data.features, test_data.targets)
                score = mean((predict_linear_baseline(model, test_data.features) .- test_data.targets).^2)
            elseif model_name == "physics_only"
                # Load physics-only model
                model = load("checkpoints/physics_only_model.jld2")
                score = evaluate_single_run(model, test_data, seed)
            else
                # Load trained neural models
                model = load("checkpoints/$(model_name)_seed_$(seed).jld2")
                score = evaluate_single_run(model, test_data, seed)
            end
            
            push!(model_scores, score)
        end
        
        results[model_name] = model_scores
    end
    
    # Statistical analysis
    println("\nPerforming statistical analysis...")
    comparisons = statistical_comparison(results)
    
    # Generate results table
    report_results_table(results, "paper/results/statistical_results.md")
    
    # Print summary
    println("\n" * "="^50)
    println("STATISTICAL EVALUATION SUMMARY")
    println("="^50)
    
    for (model, scores) in results
        ci = bootstrap_confidence_interval(scores)
        println("$model: $(round(mean(scores), digits=3)) ± $(round(std(scores), digits=3)) [$(round(ci.lower, digits=3)), $(round(ci.upper, digits=3))]")
    end
    
    println("\nPairwise Comparisons:")
    for (comparison, result) in comparisons
        significance = result.significant ? "***" : "ns"
        println("$comparison: p=$(round(result.p_value, digits=4)) $significance")
    end
    
    # Save detailed results
    save("paper/results/detailed_statistical_results.jld2", Dict(
        "results" => results,
        "comparisons" => comparisons
    ))
    
    println("\nDetailed results saved to paper/results/")
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_rigorous_evaluation()
end 