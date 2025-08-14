using CSV, DataFrames, Statistics, Random

"""
    validate_physics_discovery(ude_model, test_scenarios)

Validate whether the model actually discovered physics vs. just curve fitting.
"""
function validate_physics_discovery(ude_model, test_scenarios::Vector{String})
    println("Validating physics discovery...")

    true_params = Dict(
        "η_in" => 0.95,
        "η_out" => 0.90,
        "α" => 0.1,
        "β" => 0.05,
        "γ" => 0.02
    )

    learned_params = extract_physics_parameters(ude_model)

    param_recovery_errors = Dict{String, Float64}()
    for (param_name, true_val) in true_params
        if haskey(learned_params, param_name)
            learned_val = learned_params[param_name]
            rel_error = abs(learned_val - true_val) / max(abs(true_val), eps())
            param_recovery_errors[param_name] = rel_error
            println("$(param_name): True=$(true_val), Learned=$(round(learned_val, digits=4)), Error=$(round(rel_error*100, digits=1))%")
        else
            println("⚠️  $(param_name) not present in learned parameters")
        end
    end

    extrapolation_results = Dict{String, Float64}()
    for scenario_file in test_scenarios
        println("Testing extrapolation on: $scenario_file")
        try
            test_df = CSV.read(scenario_file, DataFrame)
            if !all([:time, :x1, :x2] .∈ propertynames(test_df))
                println("  ⚠️  Missing columns in $scenario_file; skipping")
                continue
            end
            # Build next-step labels within the file to avoid leakage
            sort!(test_df, :time)
            if nrow(test_df) < 2
                continue
            end
            test_df.x1_next = [test_df.x1[2:end]; missing]
            test_df.x2_next = [test_df.x2[2:end]; missing]
            test_df = test_df[1:end-1, :]

            preds = predict(ude_model, test_df)  # Must be implemented by the user for their model
            truth = Matrix(test_df[:, [:x1_next, :x2_next]])
            extrap_mse = mean((preds .- truth).^2)
            extrapolation_results[basename(scenario_file)] = extrap_mse
            println("  Extrapolation MSE: $(round(extrap_mse, digits=4))")
        catch e
            println("  ❌ Failed extrapolation test on $(scenario_file): $e")
        end
    end

    symbolic_validation = validate_symbolic_form(ude_model)

    return Dict(
        "parameter_recovery" => param_recovery_errors,
        "extrapolation_performance" => extrapolation_results,
        "symbolic_validation" => symbolic_validation
    )
end

"""
    extract_physics_parameters(ude_model)

Extract learned physics parameters from UDE model.
MODIFY this function based on your actual model structure.
"""
function extract_physics_parameters(ude_model)
    try
        params = ude_model.physics_params  # MODIFY this accessor for your model
        return Dict(
            "η_in" => params[1],
            "η_out" => params[2],
            "α" => params[3],
            "β" => params[4],
            "γ" => params[5]
        )
    catch e
        println("Warning: Could not extract physics parameters: $e")
        return Dict{String, Float64}()
    end
end

"""
    validate_symbolic_form(ude_model)

Validate the symbolic form discovered by the model.
"""
function validate_symbolic_form(ude_model)
    println("Validating symbolic form...")

    # Generate test inputs for symbolic validation
    test_inputs = randn(1000, 2)  # MODIFY dimensions based on your model

    try
        # Attempt to obtain neural component outputs; MODIFY for your model
        nn_outputs = ude_model.neural_component(test_inputs)

        X = create_polynomial_features(test_inputs; degree=3)
        β = X \ nn_outputs
        preds = X * β

        r_squared = calculate_r_squared(nn_outputs, preds)
        physics_generalization_score = test_physics_generalization(ude_model)

        return Dict(
            "polynomial_r_squared" => r_squared,
            "physics_generalization" => physics_generalization_score,
            "meaningful_physics" => physics_generalization_score > 0.8
        )

    catch e
        println("Warning: Symbolic validation failed: $e")
        return Dict("error" => string(e))
    end
end

function create_polynomial_features(X::AbstractMatrix; degree::Int=3)
    n, d = size(X)
    cols = [ones(n)]
    # Add degree-1 terms
    for j in 1:d
        push!(cols, X[:, j])
    end
    if degree >= 2
        # Quadratic terms
        for j in 1:d
            push!(cols, X[:, j].^2)
        end
        for j in 1:d
            for k in j+1:d
                push!(cols, X[:, j] .* X[:, k])
            end
        end
    end
    if degree >= 3
        for j in 1:d
            push!(cols, X[:, j].^3)
        end
        # Cross-degree terms (simple subset)
        for j in 1:d
            for k in j+1:d
                push!(cols, (X[:, j].^2) .* X[:, k])
                push!(cols, X[:, j] .* (X[:, k].^2))
            end
        end
    end
    return hcat(cols...)
end

function calculate_r_squared(y_true, y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0
end

function test_physics_generalization(ude_model)
    # Placeholder generalization score; replace with scenario-based tests
    return 0.0
end

# Utility to load model if needed by external scripts
function load_trained_model(path::AbstractString)
    try
        model = JLD2.load(path, "model")
        return model
    catch e
        println("Warning: Could not load model from $(path): $e")
        return nothing
    end
end 