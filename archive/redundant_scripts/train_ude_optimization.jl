#!/usr/bin/env julia

"""
    train_ude_optimization.jl

Proper UDE training using optimization-based methods (not Bayesian inference).
This is the correct approach for Universal Differential Equations.

UDE = Universal Differential Equation (hybrid physics + neural)
NOT BNODE = Bayesian Neural ODE (which would use Bayesian inference)

APPROACH:
- Use optimization (ADAM, L-BFGS) instead of MCMC
- Point estimation of parameters
- Traditional uncertainty quantification if needed
- Focus on model performance, not Bayesian uncertainty
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

println("🚀 UDE OPTIMIZATION TRAINING")
println("=" ^ 50)
println("Training Universal Differential Equation (UDE)")
println("Using optimization-based methods (NOT Bayesian inference)")
println()

# RESEARCH: Load training data
println("📊 LOADING TRAINING DATA")
println("-" ^ 30)

# Load data using existing infrastructure
data_path = joinpath(@__DIR__, "..", "data", "training_dataset.csv")
if !isfile(data_path)
    error("Training data not found at: $data_path")
end

df = CSV.read(data_path, DataFrame)
println("  → Loaded $(nrow(df)) training samples")
println("  → Features: $(names(df))")

# Prepare data
t = Array(df.time)
Y = Matrix(df[:, [:x1, :x2]])

println("  → Time range: $(minimum(t)) to $(maximum(t))")
println("  → State variables: x1 (SOC), x2 (Power)")

# RESEARCH: Define UDE model (optimization-based)
println("\n🔧 DEFINING UDE MODEL")
println("-" ^ 30)

# Physics parameters (to be optimized)
physics_params = [0.9, 0.9, 0.001, 1.0, 0.001]  # ηin, ηout, α, β, γ

# Neural network parameters (to be optimized)
nn_params = randn(15) * 0.1  # 15 neural parameters

# Combine all parameters
all_params = [physics_params; nn_params]

println("  → Physics parameters: 5 (ηin, ηout, α, β, γ)")
println("  → Neural parameters: 15")
println("  → Total parameters: $(length(all_params))")

# RESEARCH: Define neural network function
function ude_nn(x, params)
    # Extract neural parameters
    nn_weights = params[6:end]
    
    # Simple neural network: 2 inputs -> 5 hidden -> 1 output
    W1 = reshape(nn_weights[1:10], 5, 2)
    b1 = nn_weights[11:15]
    
    # Forward pass
    h = tanh.(W1 * x + b1)
    output = sum(h)
    
    return output
end

# RESEARCH: Define UDE system
function ude_system!(du, u, p, t)
    x1, x2 = u
    
    # Extract physics parameters
    ηin, ηout, α, β, γ = p[1:5]
    
    # Neural correction
    neural_correction = ude_nn([x1, x2], p)
    
    # Physics-based dynamics
    du[1] = ηin * (1 - x1) - α * x1 + neural_correction  # SOC dynamics
    du[2] = β * (x1 - 0.5) - γ * x2 + neural_correction  # Power dynamics
end

# RESEARCH: Define loss function
function ude_loss(params, t_data, Y_data)
    # Create ODE problem
    prob = ODEProblem(ude_system!, [0.5, 0.0], (minimum(t_data), maximum(t_data)), params)
    
    # Solve ODE
    sol = solve(prob, Tsit5(); saveat=t_data, abstol=1e-6, reltol=1e-6)
    
    if sol.retcode != :Success
        return Inf  # Return high loss for failed solves
    end
    
    # Extract predictions
    Y_pred = hcat([sol(t) for t in t_data]...)
    Y_pred = Y_pred'
    
    # Calculate MSE loss
    mse = mean((Y_pred - Y_data).^2)
    
    return mse
end

# RESEARCH: Optimization setup
println("\n🎯 OPTIMIZATION SETUP")
println("-" ^ 30)

# Initial parameters
initial_params = copy(all_params)

# Loss function wrapper for Optim
function loss_wrapper(params)
    return ude_loss(params, t, Y)
end

# Gradient function (using finite differences for simplicity)
function grad_wrapper(grad, params)
    ε = 1e-6
    
    for i in 1:length(params)
        params_plus = copy(params)
        params_minus = copy(params)
        params_plus[i] += ε
        params_minus[i] -= ε
        
        grad[i] = (loss_wrapper(params_plus) - loss_wrapper(params_minus)) / (2 * ε)
    end
    
    return nothing
end

println("  → Using L-BFGS optimization")
println("  → Initial loss: $(round(loss_wrapper(initial_params), digits=6))")

# RESEARCH: Run optimization
println("\n🔄 RUNNING OPTIMIZATION")
println("-" ^ 30)

# Optimization options
options = Optim.Options(
    iterations=1000,
    g_tol=1e-6,
    x_tol=1e-6,
    f_tol=1e-6,
    show_trace=true,
    show_every=50
)

# Run optimization
println("  → Starting optimization...")
result = Optim.optimize(loss_wrapper, grad_wrapper, initial_params, LBFGS(), options)

# RESEARCH: Extract results
println("\n📊 OPTIMIZATION RESULTS")
println("-" ^ 30)

optimized_params = result.minimizer
final_loss = result.minimum
converged = Optim.converged(result)

println("  → Optimization converged: $(converged ? "✅ YES" : "❌ NO")")
println("  → Final loss: $(round(final_loss, digits=6))")
println("  → Iterations: $(result.iterations)")

# RESEARCH: Extract optimized parameters
ηin_opt, ηout_opt, α_opt, β_opt, γ_opt = optimized_params[1:5]
nn_params_opt = optimized_params[6:end]

println("\n🔧 OPTIMIZED PARAMETERS")
println("-" ^ 30)
println("Physics Parameters:")
println("  ηin:  $(round(ηin_opt, digits=6))")
println("  ηout: $(round(ηout_opt, digits=6))")
println("  α:    $(round(α_opt, digits=6))")
println("  β:    $(round(β_opt, digits=6))")
println("  γ:    $(round(γ_opt, digits=6))")
println("Neural Parameters:")
println("  Mean: $(round(mean(nn_params_opt), digits=6))")
println("  Std:  $(round(std(nn_params_opt), digits=6))")

# RESEARCH: Evaluate model performance
println("\n📈 MODEL PERFORMANCE")
println("-" ^ 30)

# Solve ODE with optimized parameters
prob_final = ODEProblem(ude_system!, [0.5, 0.0], (minimum(t), maximum(t)), optimized_params)
sol_final = solve(prob_final, Tsit5(); saveat=t, abstol=1e-6, reltol=1e-6)

# Extract predictions
Y_pred = hcat([sol_final(t) for t in t]...)
Y_pred = Y_pred'

# Calculate performance metrics
rmse_x1 = sqrt(mean((Y_pred[:, 1] - Y[:, 1]).^2))
rmse_x2 = sqrt(mean((Y_pred[:, 2] - Y[:, 2]).^2))

r2_x1 = 1 - sum((Y_pred[:, 1] - Y[:, 1]).^2) / sum((Y[:, 1] .- mean(Y[:, 1])).^2)
r2_x2 = 1 - sum((Y_pred[:, 2] - Y[:, 2]).^2) / sum((Y[:, 2] .- mean(Y[:, 2])).^2)

println("Performance Metrics:")
println("  x1 (SOC):")
println("    RMSE: $(round(rmse_x1, digits=6))")
println("    R²:  $(round(r2_x1, digits=6))")
println("  x2 (Power):")
println("    RMSE: $(round(rmse_x2, digits=6))")
println("    R²:  $(round(r2_x2, digits=6))")

# RESEARCH: Save results
println("\n💾 SAVING RESULTS")
println("-" ^ 30)

results = Dict(
    :model_type => "universal_differential_equation",
    :training_method => "optimization",
    :physics_params => Dict(
        :ηin => ηin_opt,
        :ηout => ηout_opt,
        :α => α_opt,
        :β => β_opt,
        :γ => γ_opt
    ),
    :neural_params => nn_params_opt,
    :performance => Dict(
        :rmse_x1 => rmse_x1,
        :rmse_x2 => rmse_x2,
        :r2_x1 => r2_x1,
        :r2_x2 => r2_x2
    ),
    :optimization => Dict(
        :final_loss => final_loss,
        :converged => converged,
        :iterations => result.iterations
    ),
    :metadata => Dict(
        :n_samples => length(t),
        :n_physics_params => 5,
        :n_neural_params => 15,
        :training_method => "L-BFGS optimization"
    )
)

# Save to checkpoint
BSON.@save joinpath(@__DIR__, "..", "checkpoints", "ude_optimization_results.bson") results

println("  → Results saved to checkpoints/ude_optimization_results.bson")

# RESEARCH: Summary
println("\n🎯 UDE OPTIMIZATION SUMMARY")
println("=" ^ 50)
println("✅ Model Type: Universal Differential Equation (UDE)")
println("✅ Training Method: Optimization (L-BFGS)")
println("✅ Physics Parameters: 5 (ηin, ηout, α, β, γ)")
println("✅ Neural Parameters: 15")
println("✅ Convergence: $(converged ? "✅ YES" : "❌ NO")")
println("✅ Final Loss: $(round(final_loss, digits=6))")
println()
println("Performance:")
println("  x1 (SOC): R² = $(round(r2_x1, digits=4))")
println("  x2 (Power): R² = $(round(r2_x2, digits=4))")
println()
println("🎯 UDE OPTIMIZATION COMPLETED SUCCESSFULLY!")
println("=" ^ 50) 