using Statistics, HypothesisTests, StatsBase, Random, DataFrames

"""
    evaluate_model_with_stats(model, test_data, n_bootstrap=1000)

Evaluate model with proper statistical analysis including confidence intervals.
Assumes the model supports `predict(model, df)` returning an N×2 Matrix for [:x1_next, :x2_next].
"""
function evaluate_model_with_stats(model, test_data::DataFrame, n_bootstrap::Int=1000)
    println("Evaluating model with statistical analysis...")

    cols = Set(Symbol.(names(test_data)))
    if !all(x -> x in cols, [:x1_next, :x2_next])
        error("test_data must contain :x1_next and :x2_next columns")
    end

    # Robust prediction helper: try BaselineModels.predict first, then generic
    function _predict_any(m, df)
        if isdefined(Main, :BaselineModels)
            try
                return Main.BaselineModels.predict(m, df)
            catch
                # fallthrough
            end
        end
        return Base.invokelatest(predict, m, df)
    end

    predictions = nothing
    try
        predictions = _predict_any(model, test_data)
    catch e
        println("❌ Prediction failed: $e")
        rethrow(e)
    end

    true_values = Matrix(test_data[:, [:x1_next, :x2_next]])

    println("  Debug: predictions size = ", try; size(predictions); catch; "(unknown)"; end,
            ", eltype = ", try; string(eltype(predictions)); catch; "(unknown)"; end)
    println("  Debug: true_values size = ", size(true_values), ", eltype = ", string(eltype(true_values)))

    if ndims(predictions) != 2
        error("Predictions must be a 2D matrix; got ndims=$(ndims(predictions))")
    end

    if size(predictions, 1) != size(true_values, 1)
        error("Prediction length $(size(predictions,1)) does not match truth $(size(true_values,1))")
    end
    if size(predictions, 2) != 2
        error("Predictions must have 2 columns for [:x1_next, :x2_next]; got $(size(predictions,2))")
    end

    mse_per_point = vec(mean((predictions .- true_values).^2, dims=2))

    bootstrap_mse = Float64[]
    for i in 1:n_bootstrap
        idx = StatsBase.sample(1:length(mse_per_point), length(mse_per_point); replace=true)
        push!(bootstrap_mse, mean(mse_per_point[idx]))
    end

    mean_mse = mean(mse_per_point)
    ci_lower = quantile(bootstrap_mse, 0.025)
    ci_upper = quantile(bootstrap_mse, 0.975)
    std_error = std(bootstrap_mse)

    results = Dict(
        "mean_mse" => mean_mse,
        "std_error" => std_error,
        "ci_95_lower" => ci_lower,
        "ci_95_upper" => ci_upper,
        "n_samples" => length(mse_per_point)
    )

    println("MSE: $(round(mean_mse, digits=4)) ± $(round(std_error, digits=4))")
    println("95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")

    return results
end

"""
    compare_models_statistical(results_a, results_b, method_name_a, method_name_b)

Compare two models using proper statistical tests.
Accepts either per-model evaluation dicts (with keys: mean_mse, std_error, ci_95_lower/upper)
or aggregated dicts across seeds (with keys: mean_mse, std_mse, n_seeds).
"""
function compare_models_statistical(results_a::Dict, results_b::Dict, method_name_a::AbstractString, method_name_b::AbstractString)
    println("\n" * "="^50)
    println("STATISTICAL COMPARISON: $method_name_a vs $method_name_b")
    println("="^50)

    # Derive CI bounds if absent
    function ci_bounds(res::Dict)
        if haskey(res, "ci_95_lower") && haskey(res, "ci_95_upper")
            return (res["ci_95_lower"], res["ci_95_upper"])
        elseif haskey(res, "std_mse") && haskey(res, "n_seeds") && res["n_seeds"] > 1
            margin = 1.96 * res["std_mse"] / sqrt(res["n_seeds"])
            return (res["mean_mse"] - margin, res["mean_mse"] + margin)
        else
            return (NaN, NaN)
        end
    end

    (a_low, a_high) = ci_bounds(results_a)
    (b_low, b_high) = ci_bounds(results_b)

    overlap = true
    if !(isnan(a_low) || isnan(b_low))
        overlap = !(a_high < b_low || b_high < a_low)
    else
        println("⚠️  Could not compute CI overlap (missing information)")
    end

    if overlap
        println("⚠️  Confidence intervals OVERLAP - difference may not be significant")
    else
        println("✅ Confidence intervals DO NOT overlap - likely significant difference")
    end

    # Choose variability measure
    std_a = haskey(results_a, "std_error") ? results_a["std_error"] : (haskey(results_a, "std_mse") ? results_a["std_mse"] : NaN)
    std_b = haskey(results_b, "std_error") ? results_b["std_error"] : (haskey(results_b, "std_mse") ? results_b["std_mse"] : NaN)

    pooled_std = (!isnan(std_a) && !isnan(std_b)) ? sqrt((std_a^2 + std_b^2) / 2) : NaN
    cohens_d = (!isnan(pooled_std) && pooled_std > 0) ? abs(results_a["mean_mse"] - results_b["mean_mse"]) / pooled_std : 0.0

    effect_size_interpretation = cohens_d < 0.2 ? "negligible" : cohens_d < 0.5 ? "small" : cohens_d < 0.8 ? "medium" : "large"

    println("Effect size (Cohen's d): $(round(cohens_d, digits=3)) ($effect_size_interpretation)")

    return Dict(
        "confidence_intervals_overlap" => overlap,
        "cohens_d" => cohens_d,
        "effect_size" => effect_size_interpretation
    )
end 