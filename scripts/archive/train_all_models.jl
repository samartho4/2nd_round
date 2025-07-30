# Combined Model Training Script
using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

Random.seed!(42)

# Load data
function load_data(split, max_points=500)
    # Handle the training dataset name difference
    if split == "train"
        df = CSV.read("data/training_dataset.csv", DataFrame)
    else
        df = CSV.read("data/$(split)_dataset.csv", DataFrame)
    end
    if nrow(df) > max_points
        df = df[randperm(nrow(df))[1:max_points], :]
    end
    sort!(df, :time)
    t = Array(df.time)
    Y = Matrix(df[:, [:x1, :x2]])
    u0 = Y[1, :]
    return t, Y, u0
end

t_train, Y_train, u0 = load_data("train")
t_val, Y_val, u0_val = load_data("val", 200)
t_test, Y_test, u0_test = load_data("test", 200)

println("Data loaded: $(size(Y_train,1)) train, $(size(Y_val,1)) val, $(size(Y_test,1)) test points")

# Bayesian Neural ODE
@model function bayesian_node(t, Y, u0)
    σ ~ truncated(Normal(0.05, 0.05), 1e-3, 0.5)
    θ ~ MvNormal(zeros(10), 0.3)

    prob = ODEProblem(baseline_nn!, u0, (minimum(t), maximum(t)), θ)
    sol = solve(prob, Tsit5(), saveat=t, abstol=1e-5, reltol=1e-5, maxiters=5000)

    if sol.retcode != :Success || length(sol) != length(t)
        Turing.@addlogprob! -Inf
        return
    end

    Ŷ = hcat(sol.u...)'
    for i in 1:length(t)
        Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
    end
end

# UDE dynamics
function ude_dynamics!(dx, x, p, t)
    x1, x2 = x
    ηin, ηout, α, β, γ = p[1:5]
    nn_params = p[6:end]
    
    u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
    Pgen = max(0, sin((t - 6) * π / 12))
    Pload = 0.6 + 0.2 * sin(t * π / 12)
    
    Pin = u > 0 ? ηin * u : (1 / ηout) * u
    d = Pload
    dx[1] = Pin - d
    
    nn_output = simple_ude_nn([x1, x2, Pgen, Pload, t], nn_params)
    dx[2] = -α * x2 + nn_output + γ * x1
end

function simple_ude_nn(input, params)
    x1, x2, Pgen, Pload, t = input
    h1 = tanh(params[1]*x1 + params[2]*x2 + params[3]*Pgen + params[4]*Pload + params[5]*t + params[6])
    h2 = tanh(params[7]*x1 + params[8]*x2 + params[9]*Pgen + params[10]*Pload + params[11]*t + params[12])
    return params[13]*h1 + params[14]*h2 + params[15]
end

@model function bayesian_ude(t, Y, u0)
    σ ~ truncated(Normal(0.1, 0.05), 0.01, 0.5)
    
    ηin ~ truncated(Normal(0.9, 0.1), 0.5, 1.0)
    ηout ~ truncated(Normal(0.9, 0.1), 0.5, 1.0)
    α ~ truncated(Normal(0.001, 0.0005), 0.0001, 0.01)
    β ~ truncated(Normal(1.0, 0.2), 0.5, 2.0)
    γ ~ truncated(Normal(0.001, 0.0005), 0.0001, 0.01)
    
    nn_params ~ MvNormal(zeros(15), 0.1)
    
    p = [ηin, ηout, α, β, γ, nn_params...]
    
    prob = ODEProblem(ude_dynamics!, u0, (minimum(t), maximum(t)), p)
    sol = solve(prob, Tsit5(), saveat=t, abstol=1e-5, reltol=1e-5, maxiters=5000)
    
    if sol.retcode != :Success || length(sol) != length(t)
        Turing.@addlogprob! -Inf
        return
    end
    
    Ŷ = hcat(sol.u...)'
    for i in 1:length(t)
        Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
    end
end

# Training function
function train_model(model_type, t, Y, u0, n_samples=100)
    println("Training $model_type model...")
    
    if model_type == "bayesian"
        model = bayesian_node(t, Y, u0)
    elseif model_type == "ude"
        model = bayesian_ude(t, Y, u0)
    else
        error("Unknown model type: $model_type")
    end
    
    chain = sample(model, NUTS(0.65), n_samples + 20, discard_initial=20, thinning=1, progress=true)
    
    # Save results
    results = Dict(
        :chain => chain,
        :model_type => model_type,
        :n_samples => n_samples
    )
    
    BSON.@save "checkpoints/$(model_type)_models.bson" results
    println("$model_type model trained and saved")
    
    return results
end

# Train both models
if length(ARGS) > 0 && ARGS[1] == "bayesian"
    train_model("bayesian", t_train, Y_train, u0)
elseif length(ARGS) > 0 && ARGS[1] == "ude"
    train_model("ude", t_train, Y_train, u0)
else
    println("Training both models...")
    train_model("bayesian", t_train, Y_train, u0)
    train_model("ude", t_train, Y_train, u0)
end 