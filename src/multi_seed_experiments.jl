using Random
using Glob: glob

"""
    run_multi_seed_experiment(model_configs, seeds=1:10)

Run experiments with multiple random seeds for robust evaluation.
"""
function run_multi_seed_experiment(model_configs::Dict, seeds=1:10)
    println("Starting multi-seed experiments...")

    all_results = Dict{Int, Dict}()

    for seed in seeds
        println("\n" * "="^30)
        println("RUNNING SEED: $seed")
        println("="^30)

        Random.seed!(seed)

        # Create seed-specific data splits from per-scenario dense CSVs
        scenario_files = glob("data/scenarios/*/true_dense.csv")
        train, val, test = create_temporal_splits(scenario_files; seed=seed)
        train_expanded = augment_training_data(train, 15000)

        seed_results = Dict{String, Any}()

        for (config_name, config) in model_configs
            println("\nTraining: $config_name")
            try
                model = nothing
                if config_name == "UDE"
                    error("train_ude_model not implemented; add your training function")
                elseif config_name == "BNN_ODE"
                    error("train_bnn_ode_model not implemented; add your training function")
                elseif config_name == "LinearRegression"
                    model = train_baseline(LinearRegressionBaseline, train_expanded)
                elseif config_name == "RandomForest"
                    model = train_baseline(BaselineModels._RFStub, train_expanded)
                else
                    println("  ⚠️  Unknown config: $config_name. Skipping.")
                    continue
                end

                results = evaluate_model_with_stats(model, test)
                seed_results[config_name] = results
                println("✅ $config_name completed - MSE: $(round(results["mean_mse"], digits=4))")

            catch e
                println("❌ $config_name failed: $e")
                seed_results[config_name] = Dict("error" => string(e))
            end
        end

        all_results[seed] = seed_results
    end

    return all_results
end 