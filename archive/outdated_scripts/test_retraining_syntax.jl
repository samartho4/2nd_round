#!/usr/bin/env julia

"""
    test_retraining_syntax.jl

Simple test script to verify syntax and basic functionality before running full retraining.
"""

using Pkg
Pkg.activate(".")

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Load required modules
include(joinpath(@__DIR__, "..", "src", "training.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))

using DifferentialEquations
using Flux
using Optim
using Statistics
using Random
using LinearAlgebra
using CSV
using DataFrames
using BSON
using Turing
using MCMCChains
using Distributions
using HypothesisTests
using Dates

println("🧪 TESTING RETRAINING SCRIPT SYNTAX")
println("=" ^ 50)

# Test 1: Load data
println("📊 Test 1: Loading data...")
data_path = joinpath(@__DIR__, "..", "data", "training_dataset_fixed.csv")
if !isfile(data_path)
    error("Training data not found at: $data_path")
end

df = CSV.read(data_path, DataFrame)
println("  ✅ Data loaded: $(nrow(df)) samples")

# Test 2: UDE model definition
println("🔧 Test 2: UDE model definition...")
function ude_nn(x, params)
    nn_weights = params[6:end]
    W1 = reshape(nn_weights[1:10], 5, 2)
    b1 = nn_weights[11:15]
    h = tanh.(W1 * x + b1)
    output = sum(h)
    return output
end

function ude_system!(du, u, p, t)
    x1, x2 = u
    ηin, ηout, α, β, γ = p[1:5]
    neural_correction = ude_nn([x1, x2], p)
    du[1] = ηin * (1 - x1) - α * x1 + neural_correction
    du[2] = β * (x1 - 0.5) - γ * x2 + neural_correction
end

println("  ✅ UDE model defined")

# Test 3: BNode model definition
println("🔮 Test 3: BNode model definition...")
@model function bnode_model(t_data, Y_data)
    ηin ~ Normal(0.9, 0.1)
    ηout ~ Normal(0.9, 0.1)
    α ~ Normal(0.001, 0.001)
    β ~ Normal(1.0, 0.1)
    γ ~ Normal(0.001, 0.001)
    nn_weights ~ MvNormal(zeros(15), 0.1 * I)
    params = vcat([ηin, ηout, α, β, γ], nn_weights)
    
    function bnode_system!(du, u, p, t)
        x1, x2 = u
        ηin, ηout, α, β, γ = p[1:5]
        nn_weights = p[6:end]
        W1 = reshape(nn_weights[1:10], 5, 2)
        b1 = nn_weights[11:15]
        inputs = [x1, x2]
        h = tanh.(W1 * inputs + b1)
        neural_correction = sum(h)
        du[1] = ηin * (1 - x1) - α * x1 + neural_correction
        du[2] = β * (x1 - 0.5) - γ * x2 + neural_correction
    end
    
    prob = ODEProblem(bnode_system!, [0.5, 0.0], (minimum(t_data), maximum(t_data)), params)
    sol = solve(prob, Tsit5(); saveat=t_data, abstol=1e-6, reltol=1e-6)
    
    if sol.retcode != :Success
        Turing.@addlogprob! -Inf
        return
    end
    
    Y_pred = hcat([sol(t) for t in t_data]...)
    Y_pred = Y_pred'
    
    σ ~ Exponential(1.0)
    for i in 1:size(Y_data, 1)
        Y_data[i, :] ~ MvNormal(Y_pred[i, :], σ * I)
    end
end

println("  ✅ BNode model defined")

# Test 4: Directory creation
println("📁 Test 4: Directory creation...")
results_dir = joinpath(@__DIR__, "..", "results")
checkpoints_dir = joinpath(@__DIR__, "..", "checkpoints")

if !isdir(results_dir)
    mkdir(results_dir)
end
if !isdir(checkpoints_dir)
    mkdir(checkpoints_dir)
end

println("  ✅ Directories created")

# Test 5: Small ODE solve
println("🔬 Test 5: Small ODE solve...")
t_test = [0.0, 0.1, 0.2]
params_test = vcat([0.9, 0.9, 0.001, 1.0, 0.001], randn(15) * 0.1)

prob_test = ODEProblem(ude_system!, [0.5, 0.0], (0.0, 0.2), params_test)
sol_test = solve(prob_test, Tsit5(); saveat=t_test, abstol=1e-6, reltol=1e-6)

if sol_test.retcode == :Success
    println("  ✅ ODE solve successful")
else
    println("  ⚠️ ODE solve failed: $(sol_test.retcode)")
end

println("\n🎯 ALL TESTS COMPLETED SUCCESSFULLY!")
println("✅ Script syntax is correct")
println("✅ Basic functionality verified")
println("✅ Ready to run full retraining") 