module StatisticalAnalysis

using Statistics, StatsBase, Random
using HypothesisTests
export statistical_comparison, bootstrap_confidence_interval, report_results_table

function statistical_comparison(results_dict; alpha=0.05)
    """
    Compare model results with statistical significance testing
    results_dict: Dict with model_name => [list of scores from multiple runs]
    Returns: Dict with pairwise comparisons and p-values
    """
    models = collect(keys(results_dict))
    n_models = length(models)
    comparison_results = Dict()
    
    for i in 1:n_models
        for j in (i+1):n_models
            model_a, model_b = models[i], models[j]
            scores_a = results_dict[model_a]
            scores_b = results_dict[model_b]
            
            # Welch's t-test for unequal variances
            test_result = UnequalVarianceTTest(scores_a, scores_b)
            
            comparison_results["$(model_a)_vs_$(model_b)"] = (
                p_value = pvalue(test_result),
                significant = pvalue(test_result) < alpha,
                mean_diff = mean(scores_a) - mean(scores_b),
                effect_size = (mean(scores_a) - mean(scores_b)) / sqrt((var(scores_a) + var(scores_b)) / 2)
            )
        end
    end
    
    return comparison_results
end

function bootstrap_confidence_interval(data; confidence_level=0.95, n_bootstrap=1000)
    """
    Calculate bootstrap confidence intervals
    """
    n = length(data)
    bootstrap_means = []
    
    for _ in 1:n_bootstrap
        bootstrap_sample = sample(data, n, replace=true)
        push!(bootstrap_means, mean(bootstrap_sample))
    end
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (100 - alpha/2) * 100
    
    return (
        lower = percentile(bootstrap_means, lower_percentile),
        upper = percentile(bootstrap_means, upper_percentile)
    )
end

function report_results_table(results_dict, output_file)
    """
    Generate publication-ready results table with statistics
    """
    open(output_file, "w") do f
        println(f, "| Method | Mean ± Std | 95% CI | Min | Max | N Runs |")
        println(f, "|--------|------------|--------|-----|-----|--------|")
        
        for (method, scores) in results_dict
            ci = bootstrap_confidence_interval(scores)
            println(f, "| $method | $(round(mean(scores), digits=3)) ± $(round(std(scores), digits=3)) | [$(round(ci.lower, digits=3)), $(round(ci.upper, digits=3))] | $(round(minimum(scores), digits=3)) | $(round(maximum(scores), digits=3)) | $(length(scores)) |")
        end
    end
    println("Results table saved to: $output_file")
end

end  # module 