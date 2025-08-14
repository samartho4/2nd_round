module BaselineModels

using LinearAlgebra, Statistics, Random
using GLM, DataFrames
using ..StatisticalFramework

export LinearRegressionBaseline, RandomBaseline, PhysicsOnlyBaseline
export train_linear_baseline, evaluate_linear_baseline
export train_random_baseline, evaluate_random_baseline
export train_physics_baseline, evaluate_physics_baseline
# New generic interfaces
export train_baseline, predict

struct LinearRegressionBaseline
    coefficients::Vector{Float64}
    intercept::Float64
    feature_names::Vector{String}
end

struct RandomBaseline
    mean_target::Float64
    std_target::Float64
    seed::Int
end

struct PhysicsOnlyBaseline
    parameters::Dict{String, Float64}
    ode_solver_config::Dict{String, Any}
end

function train_linear_baseline(X::Matrix{Float64}, y::Vector{Float64}; feature_names=nothing)
    """Train simple linear regression baseline"""
    
    if feature_names === nothing
        feature_names = ["feature_$i" for i in 1:size(X, 2)]
    end
    
    # Add intercept term
    X_with_intercept = hcat(ones(size(X, 1)), X)
    
    # Solve normal equations: β = (X'X)^(-1) X'y
    try
        β = X_with_intercept \ y
        intercept = β[1]
        coefficients = β[2:end]
        
        return LinearRegressionBaseline(coefficients, intercept, feature_names)
    catch e
        @warn "Linear regression failed, using regularized version"
        # Ridge regression with small regularization
        λ = 1e-6
        I_reg = Matrix{Float64}(I, size(X_with_intercept, 2), size(X_with_intercept, 2))
        β = (X_with_intercept' * X_with_intercept + λ * I_reg) \ (X_with_intercept' * y)
        
        intercept = β[1]
        coefficients = β[2:end]
        
        return LinearRegressionBaseline(coefficients, intercept, feature_names)
    end
end

function evaluate_linear_baseline(model::LinearRegressionBaseline, X_test::Matrix{Float64}, y_test::Vector{Float64})
    """Evaluate linear baseline model"""
    
    predictions = X_test * model.coefficients .+ model.intercept
    mse = mean((predictions .- y_test).^2)
    mae = mean(abs.(predictions .- y_test))
    r2 = 1 - sum((y_test .- predictions).^2) / sum((y_test .- mean(y_test)).^2)
    
    return Dict(
        "mse" => mse,
        "mae" => mae,
        "r2" => r2,
        "predictions" => predictions
    )
end

function train_random_baseline(y::Vector{Float64}; seed::Int=42)
    """Train random baseline that predicts mean ± noise"""
    
    Random.seed!(seed)
    mean_target = mean(y)
    std_target = std(y)
    
    return RandomBaseline(mean_target, std_target, seed)
end

function evaluate_random_baseline(model::RandomBaseline, n_predictions::Int)
    """Evaluate random baseline"""
    
    Random.seed!(model.seed)
    predictions = randn(n_predictions) * model.std_target .+ model.mean_target
    
    # For MSE calculation, we need actual targets - this is a dummy implementation
    # In practice, you'd compare against actual test targets
    dummy_targets = fill(model.mean_target, n_predictions)
    mse = model.std_target^2  # Theoretical MSE for random predictions
    
    return Dict(
        "mse" => mse,
        "predictions" => predictions,
        "theoretical_mse" => model.std_target^2
    )
end

function train_physics_baseline(training_data; physics_params=nothing)
    """Train physics-only baseline without neural components"""
    
    # Default physics parameters (you should customize these)
    if physics_params === nothing
        physics_params = Dict(
            "eta_in" => 0.95,
            "eta_out" => 0.93,
            "alpha" => 0.1,
            "beta" => 0.05,
            "gamma" => 0.02
        )
    end
    
    solver_config = Dict(
        "abstol" => 1e-6,
        "reltol" => 1e-6,
        "maxiters" => 10000
    )
    
    return PhysicsOnlyBaseline(physics_params, solver_config)
end

function evaluate_physics_baseline(model::PhysicsOnlyBaseline, test_data)
    """Evaluate physics-only baseline"""
    
    # This is a placeholder - you need to implement actual physics simulation
    # based on your microgrid system equations
    
    @warn "Physics baseline evaluation not fully implemented - using dummy MSE"
    
    # Placeholder MSE (replace with actual physics simulation)
    dummy_mse = 0.5  # You reported Physics-Only MSE ≈ 0.16 in your results
    
    return Dict(
        "mse" => dummy_mse,
        "note" => "Placeholder implementation - integrate with your physics equations"
    )
end

# =====================
# New generic interfaces
# =====================

# Build design matrix with intercept and features [:time, :x1, :x2]
function _design_matrix(df::DataFrame)
    cols = Set(Symbol.(names(df)))
    if !all(x -> x in cols, [:time, :x1, :x2])
        error("Expected columns [:time, :x1, :x2]")
    end
    return hcat(ones(nrow(df)), Matrix(df[:, [:time, :x1, :x2]]))
end

# Train LinearRegression via DataFrame interface returning next-step for both x1,x2
function train_baseline(::Type{LinearRegressionBaseline}, train_data::DataFrame)
    cols = Set(Symbol.(names(train_data)))
    if !all(x -> x in cols, [:x1_next, :x2_next])
        error("Train data must include :x1_next and :x2_next")
    end
    X = _design_matrix(train_data)
    # Fit two independent regressions by least squares
    β1 = X \ Vector(train_data[:, :x1_next])
    β2 = X \ Vector(train_data[:, :x2_next])
    # Merge into existing type by storing slope part in coefficients, intercept separately is ambiguous for two targets.
    # We will store combined coefficients in feature_names and overload predict for this case.
    # To avoid breaking old struct, we create a wrapper storing both β vectors in feature_names as metadata.
    # But better: store as a NamedTuple inside feature_names is hacky. Instead, define a tiny new struct internally.
    return (;
        model_type = :linear_multiout,
        β1 = β1,
        β2 = β2
    )
end

function predict(model::NamedTuple{(:model_type, :β1, :β2)}, df::DataFrame)
    X = _design_matrix(df)
    y1 = X * model.β1
    y2 = X * model.β2
    return hcat(y1, y2)
end

# RandomForest stub via bagged linear models (no extra deps)
function train_baseline(::Type{RandomBaseline}, train_data::DataFrame)
    # Kept for compatibility; this trains random baseline on x1_next only
    return train_random_baseline(Vector(train_data[:, :x1_next]))
end

struct _RFStub
    betas1::Vector{Vector{Float64}}
    betas2::Vector{Vector{Float64}}
end

function train_baseline(::Type{_RFStub}, train_data::DataFrame; n_trees::Int=50, seed::Int=42)
    Random.seed!(seed)
    X = _design_matrix(train_data)
    y1 = Vector(train_data[:, :x1_next])
    y2 = Vector(train_data[:, :x2_next])
    n = size(X, 1)
    betas1 = Vector{Vector{Float64}}()
    betas2 = Vector{Vector{Float64}}()
    for t in 1:n_trees
        idx = rand(1:n, n)
        push!(betas1, X[idx, :] \ y1[idx])
        push!(betas2, X[idx, :] \ y2[idx])
    end
    return _RFStub(betas1, betas2)
end

function predict(model::_RFStub, df::DataFrame)
    X = _design_matrix(df)
    y1s = [X * β for β in model.betas1]
    y2s = [X * β for β in model.betas2]
    y1 = reduce(+, y1s) ./ length(y1s)
    y2 = reduce(+, y2s) ./ length(y2s)
    return hcat(y1, y2)
end

end  # module BaselineModels 