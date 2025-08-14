module BaselineModelsNew

using DataFrames, Statistics, Random

export LinearRegressionBaseline, RandomForestBaseline
export train_baseline, predict

struct LinearRegressionBaseline
    β1::Vector{Float64}
    β2::Vector{Float64}
end

struct RandomForestBaseline
    # Placeholder for a potential MLJ/ScikitLearn model; using simple bagging as a stub
    trees1::Vector{Vector{Float64}}
    trees2::Vector{Vector{Float64}}
    features_idx::Vector{Int}
end

function _design_matrix(df::DataFrame)
    if !all([:time, :x1, :x2] .∈ propertynames(df))
        error("Expected columns [:time, :x1, :x2]")
    end
    X = hcat(ones(nrow(df)), Matrix(df[:, [:time, :x1, :x2]]))
    return X
end

function train_baseline(::Type{LinearRegressionBaseline}, train_data::DataFrame)
    if !all([:x1_next, :x2_next] .∈ propertynames(train_data))
        error("Train data must include :x1_next and :x2_next")
    end
    X = _design_matrix(train_data)
    y1 = Vector(train_data[:, :x1_next])
    y2 = Vector(train_data[:, :x2_next])
    β1 = X \ y1
    β2 = X \ y2
    return LinearRegressionBaseline(β1, β2)
end

function predict(model::LinearRegressionBaseline, df::DataFrame)
    X = _design_matrix(df)
    y1 = X * model.β1
    y2 = X * model.β2
    return hcat(y1, y2)
end

# Simple stub RandomForest: averages of random linear models to avoid heavy deps
function train_baseline(::Type{RandomForestBaseline}, train_data::DataFrame; n_trees::Int=50, seed::Int=42)
    Random.seed!(seed)
    X = _design_matrix(train_data)
    y1 = Vector(train_data[:, :x1_next])
    y2 = Vector(train_data[:, :x2_next])
    n, d = size(X)

    trees1 = Vector{Vector{Float64}}()
    trees2 = Vector{Vector{Float64}}()
    features_idx = collect(1:d)

    for t in 1:n_trees
        idx = rand(1:n, n)  # bootstrap
        feat_mask = rand(d) .> 0.3
        if sum(feat_mask) < 2
            feat_mask[rand(1:d)] = true
            feat_mask[1] = true
        end
        cols = findall(feat_mask)
        Xt = X[idx, cols]
        β1 = Xt \ y1[idx]
        β2 = Xt \ y2[idx]
        # Store as full-length coefficient vectors aligned to cols for prediction
        β1_full = zeros(d)
        β2_full = zeros(d)
        β1_full[cols] = β1
        β2_full[cols] = β2
        push!(trees1, β1_full)
        push!(trees2, β2_full)
    end

    return RandomForestBaseline(trees1, trees2, features_idx)
end

function predict(model::RandomForestBaseline, df::DataFrame)
    X = _design_matrix(df)
    preds1 = [X * β for β in model.trees1]
    preds2 = [X * β for β in model.trees2]
    y1 = reduce(+, preds1) ./ length(preds1)
    y2 = reduce(+, preds2) ./ length(preds2)
    return hcat(y1, y2)
end

end # module 