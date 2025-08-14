module StatisticalFramework

using Statistics, StatsBase, Random, Distributions
using HypothesisTests
using CSV, DataFrames, JLD2, FileIO
using Printf

export StatisticalResults, conduct_statistical_analysis, generate_neurips_table
export bootstrap_confidence_interval, compare_methods_statistically

struct StatisticalResults
    mean::Float64
    std::Float64
    median::Float64
    ci_lower::Float64
    ci_upper::Float64
    min_val::Float64
    max_val::Float64
    n_samples::Int
end

function bootstrap_confidence_interval(data::Vector{Float64}; confidence_level=0.95, n_bootstrap=2000)
    """Bootstrap confidence interval calculation"""
    if length(data) < 2
        return (NaN, NaN)
    end
    
    bootstrap_means = Float64[]
    n = length(data)
    
    for _ in 1:n_bootstrap
        bootstrap_sample = sample(data, n, replace=true)
        push!(bootstrap_means, mean(bootstrap_sample))
    end
    
    α = 1 - confidence_level
    lower_percentile = α/2 * 100
    upper_percentile = (1 - α/2) * 100
    
    ci_lower = percentile(bootstrap_means, lower_percentile)
    ci_upper = percentile(bootstrap_means, upper_percentile)
    
    return (ci_lower, ci_upper)
end

function compute_statistical_summary(scores::Vector{Float64}; confidence_level=0.95)
    """Compute comprehensive statistical summary"""
    if isempty(scores)
        return StatisticalResults(NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0)
    end
    
    μ = mean(scores)
    σ = std(scores)
    med = median(scores)
    min_val = minimum(scores)
    max_val = maximum(scores)
    n = length(scores)
    
    ci_lower, ci_upper = bootstrap_confidence_interval(scores; confidence_level=confidence_level)
    
    return StatisticalResults(μ, σ, med, ci_lower, ci_upper, min_val, max_val, n)
end

function compare_methods_statistically(method_a_scores::Vector{Float64}, method_b_scores::Vector{Float64}; method_a_name="A", method_b_name="B")
    """Perform statistical comparison between two methods"""
    
    # Welch's t-test (unequal variances assumed)
    test_result = UnequalVarianceTTest(method_a_scores, method_b_scores)
    p_value = pvalue(test_result)
    
    # Effect size (Cohen's d)
    pooled_std = sqrt((var(method_a_scores) + var(method_b_scores)) / 2)
    cohens_d = (mean(method_a_scores) - mean(method_b_scores)) / pooled_std
    
    # Practical significance
    mean_diff = mean(method_a_scores) - mean(method_b_scores)
    
    return Dict(
        "comparison" => "$(method_a_name)_vs_$(method_b_name)",
        "p_value" => p_value,
        "significant" => p_value < 0.05,
        "cohens_d" => cohens_d,
        "mean_difference" => mean_diff,
        "n_a" => length(method_a_scores),
        "n_b" => length(method_b_scores)
    )
end

function conduct_statistical_analysis(results_dict::Dict{String, Vector{Float64}})
    """Conduct comprehensive statistical analysis"""
    
    println("🔬 Conducting Statistical Analysis...")
    
    # Compute statistics for each method
    method_stats = Dict{String, StatisticalResults}()
    for (method, scores) in results_dict
        method_stats[method] = compute_statistical_summary(scores)
        println("   $method: μ=$(round(method_stats[method].mean, digits=3)) ± $(round(method_stats[method].std, digits=3))")
    end
    
    # Pairwise comparisons
    comparisons = Dict()
    method_names = collect(keys(results_dict))
    
    for i in 1:length(method_names)
        for j in (i+1):length(method_names)
            method_a, method_b = method_names[i], method_names[j]
            comparison = compare_methods_statistically(
                results_dict[method_a], 
                results_dict[method_b];
                method_a_name=method_a,
                method_b_name=method_b
            )
            comparisons[comparison["comparison"]] = comparison
        end
    end
    
    return method_stats, comparisons
end

function generate_neurips_table(method_stats::Dict{String, StatisticalResults}, comparisons::Dict, output_file::String)
    """Generate publication-ready results table"""
    
    open(output_file, "w") do f
        println(f, "# NeurIPS 2025 Statistical Results")
        println(f, "")
        println(f, "Generated on: $(Dates.now())")
        println(f, "")
        println(f, "## Model Performance Comparison (Trajectory MSE)")
        println(f, "")
        println(f, "| Method | Mean ± Std | 95% CI | Median | Range | N |")
        println(f, "|--------|------------|--------|--------|-------|---|")
        
        # Sort methods by performance (ascending MSE)
        sorted_methods = sort(collect(method_stats), by = x -> x[2].mean)
        
        for (method, stats) in sorted_methods
            μ_str = @sprintf("%.3f", stats.mean)
            σ_str = @sprintf("%.3f", stats.std)
            ci_str = @sprintf("[%.3f, %.3f]", stats.ci_lower, stats.ci_upper)
            med_str = @sprintf("%.3f", stats.median)
            range_str = @sprintf("[%.3f, %.3f]", stats.min_val, stats.max_val)
            
            println(f, "| $method | $μ_str ± $σ_str | $ci_str | $med_str | $range_str | $(stats.n_samples) |")
        end
        
        println(f, "")
        println(f, "## Statistical Significance Tests")
        println(f, "")
        println(f, "| Comparison | p-value | Significant | Cohen's d | Interpretation |")
        println(f, "|------------|---------|-------------|-----------|----------------|")
        
        for (comp_name, comp_results) in comparisons
            p_str = @sprintf("%.4f", comp_results["p_value"])
            sig_str = comp_results["significant"] ? "***" : "ns"
            d_str = @sprintf("%.3f", comp_results["cohens_d"])
            
            # Effect size interpretation
            abs_d = abs(comp_results["cohens_d"])
            if abs_d < 0.2
                effect_interp = "negligible"
            elseif abs_d < 0.5
                effect_interp = "small"
            elseif abs_d < 0.8
                effect_interp = "medium"
            else
                effect_interp = "large"
            end
            
            println(f, "| $comp_name | $p_str | $sig_str | $d_str | $effect_interp |")
        end
        
        println(f, "")
        println(f, "***: p < 0.05 (statistically significant)")
        println(f, "ns: not significant")
        println(f, "")
        println(f, "## Interpretation")
        println(f, "- Results based on $(length(collect(values(method_stats))[1] |> x -> 1:x.n_samples)) independent runs with different random seeds")
        println(f, "- 95% confidence intervals computed via bootstrap (n=2000)")
        println(f, "- Statistical significance tested using Welch's t-test")
        println(f, "- Effect sizes reported as Cohen's d")
    end
    
    println("📊 Results table saved to: $output_file")
end

end  # module StatisticalFramework 