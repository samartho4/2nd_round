using Statistics

"""
    aggregate_multi_seed_results(multi_seed_results)

Aggregate results across multiple seeds and create publication-ready table.
"""
function aggregate_multi_seed_results(multi_seed_results::Dict)
    println("Aggregating multi-seed results...")

    if isempty(multi_seed_results)
        error("No multi-seed results provided")
    end

    first_seed_results = first(values(multi_seed_results))
    method_names = [k for k in keys(first_seed_results) if !(haskey(first_seed_results[k], "error"))]

    aggregated = Dict{String, Dict}()

    for method in method_names
        mse_values = Float64[]
        for (seed, results) in multi_seed_results
            if haskey(results, method) && !haskey(results[method], "error")
                push!(mse_values, results[method]["mean_mse"])
            end
        end

        if !isempty(mse_values)
            aggregated[method] = Dict(
                "mean_mse" => mean(mse_values),
                "std_mse" => std(mse_values),
                "n_seeds" => length(mse_values),
                "all_values" => mse_values
            )
        end
    end

    return aggregated
end

"""
    create_results_table(aggregated_results)

Create a publication-ready results table.
"""
function create_results_table(aggregated_results::Dict)
    println("\n" * "="^60)
    println("FINAL RESULTS TABLE (Multi-Seed Aggregated)")
    println("="^60)

    sorted_methods = sort(collect(keys(aggregated_results)), by=method -> aggregated_results[method]["mean_mse"]) 

    println("Method                | Mean MSE ± Std    | N Seeds | 95% CI")
    println("-"^60)

    table_data = []

    for method in sorted_methods
        results = aggregated_results[method]
        mean_mse = results["mean_mse"]
        std_mse = results["std_mse"]
        n_seeds = results["n_seeds"]

        ci_str = "N/A"
        ci_lower = NaN
        ci_upper = NaN
        if n_seeds > 1
            t_critical = 2.0
            margin_error = t_critical * std_mse / sqrt(n_seeds)
            ci_lower = mean_mse - margin_error
            ci_upper = mean_mse + margin_error
            ci_str = "[$(round(ci_lower, digits=3)), $(round(ci_upper, digits=3))]"
        end

        println("$(rpad(method, 20)) | $(rpad(string(round(mean_mse, digits=3)) * " ± " * string(round(std_mse, digits=3)), 16)) | $(rpad(n_seeds, 7)) | $ci_str")

        push!(table_data, Dict(
            "method" => method,
            "mean_mse" => mean_mse,
            "std_mse" => std_mse,
            "n_seeds" => n_seeds,
            "ci_lower" => ci_lower,
            "ci_upper" => ci_upper
        ))
    end

    return table_data
end 