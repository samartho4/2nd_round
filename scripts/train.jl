# Fixed Training Script - Actually implements the 3 objectives
using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

Random.seed!(42)

println("FIXED TRAINING - IMPLEMENTING THE 3 OBJECTIVES")
println("="^60)

# Load data
df_train = CSV.read("data/training_dataset.csv", DataFrame)
df_test = CSV.read("data/test_dataset.csv", DataFrame)

# Use larger subset for better training, but not too large to avoid ODE solver issues
subset_size = 2000
df_train_subset = df_train[1:subset_size, :]
df_test_subset = df_test[1:min(300, nrow(df_test)), :]

t_train = Array(df_train_subset.time)
Y_train = Matrix(df_train_subset[:, [:x1, :x2]])
u0_train = Y_train[1, :]

t_test = Array(df_test_subset.time)
Y_test = Matrix(df_test_subset[:, [:x1, :x2]])
u0_test = Y_test[1, :]

println("Data loaded: $(size(Y_train,1)) train, $(size(Y_test,1)) test points")

# ============================================================================
# OBJECTIVE 1: Bayesian Neural ODE (Replace full ODE with neural network)
# ============================================================================
println("\n1. IMPLEMENTING BAYESIAN NEURAL ODE")
println("-"^40)

@model function bayesian_neural_ode(t, Y, u0)
    # Observation noise
    σ ~ truncated(Normal(0.1, 0.05), 0.01, 0.5)
    
    # Neural network parameters (10 parameters)
    θ ~ MvNormal(zeros(10), 0.3)
    
    # ODE solution
    prob = ODEProblem(baseline_nn!, u0, (minimum(t), maximum(t)), θ)
    sol = solve(prob, Tsit5(), saveat=t, abstol=1e-3, reltol=1e-3, maxiters=10000)
    
    if sol.retcode != :Success || length(sol) != length(t)
        Turing.@addlogprob! -Inf
        return
    end
    
    Ŷ = hcat(sol.u...)'
    for i in 1:length(t)
        Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
    end
end

# Train Bayesian Neural ODE
println("Training Bayesian Neural ODE...")
bayesian_model = bayesian_neural_ode(t_train, Y_train, u0_train)
bayesian_chain = sample(bayesian_model, NUTS(0.65), 1000, discard_initial=20, progress=true)

# Extract results
bayesian_params = Array(bayesian_chain)[:, 1:10]
bayesian_noise = Array(bayesian_chain)[:, 11]

# Save results properly
bayesian_results = Dict(
    :params_mean => mean(bayesian_params, dims=1)[1, :],
    :params_std => std(bayesian_params, dims=1)[1, :],
    :noise_mean => mean(bayesian_noise),
    :noise_std => std(bayesian_noise),
    :n_samples => size(bayesian_params, 1),
    :model_type => "bayesian_neural_ode"
)

BSON.@save "checkpoints/bayesian_neural_ode_results.bson" bayesian_results
println("✅ Bayesian Neural ODE trained and saved")

# ============================================================================
# OBJECTIVE 2: UDE (Replace only nonlinear term with neural network)
# ============================================================================
println("\n2. IMPLEMENTING UDE (Universal Differential Equations)")
println("-"^40)

# UDE dynamics: physics + neural network for nonlinear term
function ude_dynamics!(dx, x, p, t)
    x1, x2 = x
    ηin, ηout, α, β, γ = p[1:5]  # Physics parameters
    nn_params = p[6:end]          # Neural parameters (15)
    
    # Control and power inputs (same as original)
    u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
    Pgen = max(0, sin((t - 6) * π / 12))
    Pload = 0.6 + 0.2 * sin(t * π / 12)
    
    # Energy storage dynamics (physics only)
    Pin = u > 0 ? ηin * u : (1 / ηout) * u
    d = Pload
    dx[1] = Pin - d
    
    # Grid dynamics: physics + neural network for nonlinear term
    # Original: dx[2] = -α * x2 + β * (Pgen - Pload) + γ * x1
    # UDE: Replace β * (Pgen - Pload) with neural network
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
    # Observation noise
    σ ~ truncated(Normal(0.1, 0.05), 0.01, 0.5)
    
    # Physics parameters (5 parameters)
    ηin ~ truncated(Normal(0.9, 0.1), 0.5, 1.0)
    ηout ~ truncated(Normal(0.9, 0.1), 0.5, 1.0)
    α ~ truncated(Normal(0.001, 0.0005), 0.0001, 0.01)
    β ~ truncated(Normal(1.0, 0.2), 0.5, 2.0)
    γ ~ truncated(Normal(0.001, 0.0005), 0.0001, 0.01)
    
    # Neural network parameters (15 parameters)
    nn_params ~ MvNormal(zeros(15), 0.1)
    
    # Combine physics + neural parameters
    p = [ηin, ηout, α, β, γ, nn_params...]
    
    # UDE solution
    prob = ODEProblem(ude_dynamics!, u0, (minimum(t), maximum(t)), p)
    sol = solve(prob, Tsit5(), saveat=t, abstol=1e-3, reltol=1e-3, maxiters=10000)
    
    if sol.retcode != :Success || length(sol) != length(t)
        Turing.@addlogprob! -Inf
        return
    end
    
    Ŷ = hcat(sol.u...)'
    for i in 1:length(t)
        Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
    end
end

# Train UDE
println("Training UDE...")
ude_model = bayesian_ude(t_train, Y_train, u0_train)
ude_chain = sample(ude_model, NUTS(0.65), 1000, discard_initial=20, progress=true)

# Extract results
ude_params = Array(ude_chain)
physics_params = ude_params[:, 1:5]  # ηin, ηout, α, β, γ
neural_params = ude_params[:, 6:20]  # 15 neural parameters
ude_noise = ude_params[:, 21]

# Save results properly
ude_results = Dict(
    :physics_params_mean => mean(physics_params, dims=1)[1, :],
    :physics_params_std => std(physics_params, dims=1)[1, :],
    :neural_params_mean => mean(neural_params, dims=1)[1, :],
    :neural_params_std => std(neural_params, dims=1)[1, :],
    :noise_mean => mean(ude_noise),
    :noise_std => std(ude_noise),
    :n_samples => size(ude_params, 1),
    :model_type => "universal_differential_equation"
)

BSON.@save "checkpoints/ude_results_fixed.bson" ude_results
println("✅ UDE trained and saved")

# ============================================================================
# OBJECTIVE 3: Symbolic Extraction (Extract symbolic form from UDE neural network)
# ============================================================================
println("\n3. IMPLEMENTING SYMBOLIC EXTRACTION FROM UDE NEURAL NETWORK")
println("-"^40)

# Use the trained UDE neural network parameters to extract symbolic form
println("Extracting symbolic form from UDE neural network component...")

# Get neural network parameters from UDE model
ude_nn_params = ude_results[:neural_params_mean]

# Generate data points for symbolic regression
n_points = 200
x1_range = range(-10.0, 10.0, length=8)
x2_range = range(-10.0, 10.0, length=8)
Pgen_range = range(0.0, 1.0, length=5)
Pload_range = range(0.4, 0.8, length=5)
t_range = range(0.0, 24.0, length=5)

# Generate grid of points for UDE neural network inputs
symbolic_data = []
for x1 in x1_range
    for x2 in x2_range
        for Pgen in Pgen_range
            for Pload in Pload_range
                for t in t_range
                    push!(symbolic_data, [x1, x2, Pgen, Pload, t])
                end
            end
        end
    end
end

# Limit to reasonable number
symbolic_data = symbolic_data[1:min(n_points, length(symbolic_data))]

# Evaluate UDE neural network on these points
nn_outputs = []
for point in symbolic_data
    x1, x2, Pgen, Pload, t = point
    nn_output = simple_ude_nn([x1, x2, Pgen, Pload, t], ude_nn_params)
    push!(nn_outputs, Float64(nn_output))
end

# Polynomial regression for symbolic extraction of UDE neural network
println("Performing symbolic regression on UDE neural network...")

# Create feature matrix for polynomial regression with 5 inputs
function create_ude_features(x1, x2, Pgen, Pload, t)
    return [x1, x2, Pgen, Pload, t, 
            x1^2, x2^2, Pgen^2, Pload^2, t^2,
            x1*x2, x1*Pgen, x1*Pload, x1*t,
            x2*Pgen, x2*Pload, x2*t,
            Pgen*Pload, Pgen*t, Pload*t]
end

# Create feature matrix
feature_matrix = []
for (i, point) in enumerate(symbolic_data)
    features = create_ude_features(point[1], point[2], point[3], point[4], point[5])
    push!(feature_matrix, features)
end

feature_matrix = hcat(feature_matrix...)'
feature_matrix = convert(Matrix{Float64}, feature_matrix)

# Fit polynomial regression to UDE neural network output
nn_outputs = convert(Vector{Float64}, nn_outputs)
coefficients_ude_nn = feature_matrix \ nn_outputs

# Calculate R² for symbolic extraction
pred_nn_output = feature_matrix * coefficients_ude_nn
actual_nn_output = nn_outputs

r2_ude_nn = 1 - sum((pred_nn_output .- actual_nn_output).^2) / sum((actual_nn_output .- mean(actual_nn_output)).^2)

# Save symbolic extraction results for UDE neural network
symbolic_ude_results = Dict(
    :coefficients_ude_nn => coefficients_ude_nn,
    :r2_ude_nn => r2_ude_nn,
    :feature_names => ["x1", "x2", "Pgen", "Pload", "t", 
                      "x1^2", "x2^2", "Pgen^2", "Pload^2", "t^2",
                      "x1*x2", "x1*Pgen", "x1*Pload", "x1*t",
                      "x2*Pgen", "x2*Pload", "x2*t",
                      "Pgen*Pload", "Pgen*t", "Pload*t"],
    :model_type => "symbolic_ude_extraction",
    :ude_nn_params => ude_nn_params,
    :n_features => 20
)

BSON.@save "checkpoints/symbolic_ude_extraction.bson" symbolic_ude_results
println("✅ Symbolic extraction from UDE neural network completed")
println("   - R² for UDE neural network: $(round(r2_ude_nn, digits=4))")
println("   - Features: $(length(symbolic_ude_results[:feature_names])) polynomial terms")
println("   - Target: β * (Pgen - Pload) approximation")

# ============================================================================
# FINAL RESULTS SUMMARY
# ============================================================================
println("\n" * "="^60)
println("FINAL RESULTS - ALL 3 OBJECTIVES IMPLEMENTED")
println("="^60)

println("✅ OBJECTIVE 1: Bayesian Neural ODE")
println("   - Replaced full ODE with neural network")
println("   - Uncertainty quantification: $(bayesian_results[:n_samples]) samples")
println("   - Parameters: $(length(bayesian_results[:params_mean])) neural parameters")

println("\n✅ OBJECTIVE 2: UDE (Universal Differential Equations)")
println("   - Hybrid physics + neural network approach")
println("   - Physics parameters: ηin, ηout, α, β, γ (5 parameters)")
println("   - Neural parameters: 15 additional parameters")
println("   - Replaced nonlinear term β·(Pgen-Pload) with neural network")

println("\n✅ OBJECTIVE 3: Symbolic Extraction from UDE Neural Network")
println("   - Extracted symbolic form from UDE neural network component")
println("   - Polynomial regression: 20 features (x1, x2, Pgen, Pload, t)")
println("   - R² = $(round(r2_ude_nn, digits=4))")
println("   - Target: β * (Pgen - Pload) approximation")

println("\nALL 3 OBJECTIVES SUCCESSFULLY IMPLEMENTED! 🎯") 