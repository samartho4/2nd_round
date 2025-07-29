# Working Universal Differential Equation (UDE) Implementation
# Replaces only the nonlinear term β · Pgen(t) with a neural network

using DifferentialEquations, Turing, Distributions, CSV, DataFrames, Plots, Statistics
using Random, LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

println("Implementing Working Universal Differential Equation (UDE)")
println("Objective: Replace only nonlinear term β · Pgen(t) with neural network")

# Load improved data
println("Loading improved multi-scenario data...")
df_train = CSV.read("data/train_improved.csv", DataFrame)
df_val   = CSV.read("data/val_improved.csv", DataFrame)

# Use manageable subset of data for UDE training
println("Using subset of improved data for UDE training...")
subset_size = 800
df_train_subset = df_train[1:min(subset_size, nrow(df_train)), :]
df_val_subset = df_val[1:min(200, nrow(df_val)), :]

# Extract data
t_train = df_train_subset.time
Y_train = Matrix(df_train_subset[:, [:x1, :x2]])
u0_train = Y_train[1, :]

t_val = df_val_subset.time
Y_val = Matrix(df_val_subset[:, [:x1, :x2]])
u0_val = Y_val[1, :]

println("Data loaded successfully")
println("Training data: $(size(Y_train, 1)) points")
println("Validation data: $(size(Y_val, 1)) points")

# Define the UDE dynamics
function ude_dynamics!(dx, x, p, t)
    x1, x2 = x
    
    # Extract physics parameters (first 5 parameters)
    ηin, ηout, α, β, γ = p[1:5]
    
    # Extract neural network parameters (remaining parameters)
    nn_params = p[6:end]
    
    # Control input (same as original)
    u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
    
    # Power generation (same as original)
    Pgen = max(0, sin((t - 6) * π / 12))
    
    # Power load (same as original)
    Pload = 0.6 + 0.2 * sin(t * π / 12)
    
    # Energy storage dynamics (unchanged - physics only)
    Pin = u > 0 ? ηin * u : (1 / ηout) * u
    d = Pload  # Power demand from storage
    dx[1] = Pin - d
    
    # Grid power flow dynamics (UDE approach)
    # Original: dx[2] = -α * x2 + β * (Pgen - Pload) + γ * x1
    # UDE: Replace β * (Pgen - Pload) with neural network
    
    # Simple neural network for the nonlinear term
    nn_output = simple_ude_nn([x1, x2, Pgen, Pload, t], nn_params)
    
    # UDE dynamics: physics + neural network
    dx[2] = -α * x2 + nn_output + γ * x1
end

# Simple neural network for the nonlinear term
function simple_ude_nn(input, params)
    # Very simple neural network: 5 inputs → 2 hidden → 1 output
    # Total parameters: 5*2 + 2 + 2*1 + 1 = 10 + 2 + 2 + 1 = 15 parameters
    
    x1, x2, Pgen, Pload, t = input
    
    # Hidden layer (5 inputs → 2 hidden)
    h1 = tanh(params[1]*x1 + params[2]*x2 + params[3]*Pgen + params[4]*Pload + params[5]*t + params[6])
    h2 = tanh(params[7]*x1 + params[8]*x2 + params[9]*Pgen + params[10]*Pload + params[11]*t + params[12])
    
    # Output layer (2 hidden → 1 output)
    output = params[13]*h1 + params[14]*h2 + params[15]
    
    return output
end

# Working Bayesian UDE model
@model function working_bayesian_ude(t, Y, u0)
    # Observation noise
    σ ~ truncated(Normal(0.1, 0.05), 0.01, 0.5)
    
    # Physics parameters (with best parameters as priors)
    ηin ~ truncated(Normal(0.9, 0.1), 0.5, 1.0)
    ηout ~ truncated(Normal(0.9, 0.1), 0.5, 1.0)
    α ~ truncated(Normal(0.001, 0.0005), 0.0001, 0.01)
    β ~ truncated(Normal(1.0, 0.2), 0.5, 2.0)
    γ ~ truncated(Normal(0.001, 0.0005), 0.0001, 0.01)
    
    # Neural network parameters (15 parameters with small priors)
    nn_params ~ MvNormal(zeros(15), 0.1 * I(15))
    
    # Combine physics and neural network parameters
    θ = [ηin, ηout, α, β, γ, nn_params...]
    
    # Solve UDE
    prob = ODEProblem(ude_dynamics!, u0, (minimum(t), maximum(t)), θ)
    sol = solve(prob, Tsit5(), saveat=t, abstol=1e-4, reltol=1e-4, maxiters=5000)
    
    if sol.retcode != :Success
        Turing.@addlogprob! -Inf
        return
    end
    
    # Predictions
    Yhat = hcat(sol.u...)'
    
    # Ensure Yhat has the same size as Y
    if size(Yhat, 1) != size(Y, 1)
        Turing.@addlogprob! -Inf
        return
    end
    
    # Likelihood
    for i in 1:size(Y, 1)
        Y[i, :] ~ MvNormal(Yhat[i, :], σ * I(2))
    end
end

println("Training Working Bayesian UDE model...")

# Train the UDE model with more samples for better results
model = working_bayesian_ude(t_train, Y_train, u0_train)
chain = sample(model, NUTS(0.65), 520; progress=true, discard_initial=20)

println("UDE training completed!")
println("Chain summary:")
println("   - Total samples: $(length(chain))")
println("   - Physics parameters: 5")
println("   - Neural network parameters: 15")
println("   - Total parameters: 20")

# Save UDE results
using BSON
BSON.bson("checkpoints/ude_working_results.bson", 
    chain=chain, 
    model_info="Working Universal Differential Equation with 20 parameters (5 physics + 15 neural network)"
)

println("UDE results saved to checkpoints/ude_working_results.bson")

# Evaluate UDE performance
println("Evaluating UDE performance...")

# Get posterior samples
posterior_samples = Array(chain)
num_samples = size(posterior_samples, 1)

# Evaluate on validation set with more samples
val_predictions = []
for i in 1:min(100, num_samples)  # Use more samples for better evaluation
    params = posterior_samples[i, :]
    
    prob = ODEProblem(ude_dynamics!, u0_val, (minimum(t_val), maximum(t_val)), params)
    sol = solve(prob, Tsit5(), saveat=t_val, abstol=1e-4, reltol=1e-4)
    
    if sol.retcode == :Success
        Yhat = hcat(sol.u...)'
        if size(Yhat, 1) == size(Y_val, 1)
            push!(val_predictions, Yhat)
        end
    end
end

# Calculate validation metrics
if !isempty(val_predictions)
    val_mean = mean(val_predictions)
    val_std = std(val_predictions)
    
    # Calculate metrics
    mse_val = mean((Y_val .- val_mean).^2)
    mae_val = mean(abs.(Y_val .- val_mean))
    rmse_val = sqrt(mse_val)
    r2_val = 1 - sum((Y_val .- val_mean).^2) / sum((Y_val .- mean(Y_val, dims=1)).^2)
    
    println("UDE Validation Performance:")
    println("   - MSE: $(round(mse_val, digits=4))")
    println("   - MAE: $(round(mae_val, digits=4))")
    println("   - RMSE: $(round(rmse_val, digits=4))")
    println("   - R²: $(round(r2_val, digits=4))")
else
    println("UDE evaluation failed - no successful predictions")
end

# Plot UDE results
println("Generating UDE plots...")

if !isempty(val_predictions)
    # Plot validation predictions with uncertainty
    p1 = plot(t_val, Y_val[:, 1], label="True x1", color=:blue, linewidth=2)
    plot!(t_val, val_mean[:, 1], label="UDE x1", color=:red, linewidth=2)
    plot!(t_val, val_mean[:, 1] .+ 2*val_std[:, 1], label="", color=:red, alpha=0.3)
    plot!(t_val, val_mean[:, 1] .- 2*val_std[:, 1], label="", color=:red, alpha=0.3, fillrange=val_mean[:, 1] .+ 2*val_std[:, 1], fillalpha=0.3)
    title!("UDE: Energy Storage (x1)")
    xlabel!("Time")
    ylabel!("State")
    
    p2 = plot(t_val, Y_val[:, 2], label="True x2", color=:blue, linewidth=2)
    plot!(t_val, val_mean[:, 2], label="UDE x2", color=:red, linewidth=2)
    plot!(t_val, val_mean[:, 2] .+ 2*val_std[:, 2], label="", color=:red, alpha=0.3)
    plot!(t_val, val_mean[:, 2] .- 2*val_std[:, 2], label="", color=:red, alpha=0.3, fillrange=val_mean[:, 2] .+ 2*val_std[:, 2], fillalpha=0.3)
    title!("UDE: Grid Power Flow (x2)")
    xlabel!("Time")
    ylabel!("State")
    
    p_combined = plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig(p_combined, "paper/figures/ude_working_validation.png")
    
    println("UDE plots saved to paper/figures/ude_working_validation.png")
end

# Print physics parameters from UDE
println("UDE Physics Parameters (mean):")
println("   - ηin (charging efficiency): $(round(mean(chain[:ηin]), digits=3))")
println("   - ηout (discharging efficiency): $(round(mean(chain[:ηout]), digits=3))")
println("   - α (damping factor): $(round(mean(chain[:α]), digits=6))")
println("   - β (power mismatch gain): $(round(mean(chain[:β]), digits=3))")
println("   - γ (coupling coefficient): $(round(mean(chain[:γ]), digits=6))")

println("Working UDE Implementation: COMPLETED!") 