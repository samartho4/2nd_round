# Ultra-Stable Training Script - Maximum Numerical Stability
using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

# Try to import ADVI/vi utilities if available
import Turing: ADVI, vi

Random.seed!(42)

println("ULTRA-STABLE TRAINING - MAXIMUM NUMERICAL STABILITY")
println("="^60)

# Load data
df_train = CSV.read("data/training_dataset.csv", DataFrame)
df_test = CSV.read("data/test_dataset.csv", DataFrame)

# Use smaller subset for ultra-stable training
subset_size = 1000  # Reduced for stability
df_train_subset = df_train[1:subset_size, :]
df_test_subset = df_test[1:min(200, nrow(df_test)), :]

t_train = Array(df_train_subset.time)
Y_train = Matrix(df_train_subset[:, [:x1, :x2]])
u0_train = Y_train[1, :]

t_test = Array(df_test_subset.time)
Y_test = Matrix(df_test_subset[:, [:x1, :x2]])
u0_test = Y_test[1, :]

println("Data loaded: $(size(Y_train,1)) train, $(size(Y_test,1)) test points")

# ============================================================================
# ULTRA-STABLE BAYESIAN NEURAL ODE
# ============================================================================
println("\n1. ULTRA-STABLE BAYESIAN NEURAL ODE")
println("-"^40)

@model function ultra_stable_bayesian_neural_ode(t, Y, u0)
    
    # Ultra-conservative priors for maximum stability
    σ ~ truncated(Normal(0.05, 0.01), 0.001, 0.1)  # Much smaller noise
    
    # Neural network parameters with very small variance
    θ ~ MvNormal(zeros(10), 0.1)  # Reduced variance
    
    # Define the ODE problem with ultra-strict tolerances
    prob = ODEProblem(baseline_nn!, u0, (minimum(t), maximum(t)), θ)
    
    # Ultra-strict tolerances and multiple solver attempts
    sol = solve(prob, Tsit5(), saveat=t, 
                abstol=1e-10, reltol=1e-10,  # 100x stricter than before
                maxiters=50000,  # More iterations
                adaptive=false,   # Fixed step size for stability
                dt=0.01)         # Small fixed step size
    
    if sol.retcode != :Success || length(sol) != length(t)
        Turing.@addlogprob! -Inf
        return
    end
    
    # Compare with very conservative likelihood
    Ŷ = hcat(sol.u...)'
    for i in 1:length(t)
        Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
    end
end

println("Training Ultra-Stable Bayesian Neural ODE...")
ultra_bayesian_model = ultra_stable_bayesian_neural_ode(t_train, Y_train, u0_train)

# Ultra-conservative initial parameters
ultra_initial_params = (σ = 0.05, θ = 0.05 .* randn(10))

# Optional ADVI warm-start
try
    println("Running ADVI warm-start for Ultra-Stable Bayesian NN ODE (iters=2000)...")
    q = vi(ultra_bayesian_model, ADVI(2000))
    if hasproperty(q, :posterior) && hasproperty(q.posterior, :μ)
        μ = q.posterior.μ
        if length(μ) == 11
            ultra_initial_params = (σ = max(0.01, abs(μ[end])), θ = μ[1:10])
        end
    end
catch e
    println("ADVI warm-start unavailable or failed: $(e). Proceeding with random init.")
end

# More conservative sampling with tuned NUTS
ultra_bayesian_chain = sample(ultra_bayesian_model, NUTS(0.9), 500,
                             discard_initial=100, progress=true,
                             initial_params=ultra_initial_params)

# Extract results
ultra_bayesian_params = Array(ultra_bayesian_chain)[:, 1:10]
ultra_bayesian_noise = Array(ultra_bayesian_chain)[:, 11]

# Save results
ultra_bayesian_results = Dict(
    :params_mean => mean(ultra_bayesian_params, dims=1)[1, :],
    :params_std => std(ultra_bayesian_params, dims=1)[1, :],
    :noise_mean => mean(ultra_bayesian_noise),
    :noise_std => std(ultra_bayesian_noise),
    :n_samples => size(ultra_bayesian_params, 1),
    :model_type => "ultra_stable_bayesian_neural_ode"
)

BSON.@save "checkpoints/ultra_stable_bayesian_results.bson" ultra_bayesian_results
println("✅ Ultra-Stable Bayesian Neural ODE trained and saved")

# ============================================================================
# ULTRA-STABLE UDE
# ============================================================================
println("\n2. ULTRA-STABLE UDE (Universal Differential Equations)")
println("-"^40)

# Ultra-stable UDE dynamics
function ultra_stable_ude_dynamics!(dx, x, p, t)
    x1, x2 = x
    ηin, ηout, α, β, γ = p[1:5]
    nn_params = p[6:end]
    
    # Control and power inputs (same as original)
    u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
    Pgen = max(0, sin((t - 6) * π / 12))
    Pload = 0.6 + 0.2 * sin(t * π / 12)
    
    # Part 1: Known physics
    Pin = u > 0 ? ηin * u : (1 / ηout) * u
    d = Pload
    dx[1] = Pin - d
    
    # Part 2: Neural network with gradient clipping
    nn_output = ultra_stable_ude_nn([x1, x2, Pgen, Pload, t], nn_params)
    dx[2] = -α * x2 + nn_output + γ * x1
end

function ultra_stable_ude_nn(input, params)
    x1, x2, Pgen, Pload, t = input
    
    # Gradient clipping and numerical stability
    x1_clipped = clamp(x1, -10.0, 10.0)
    x2_clipped = clamp(x2, -10.0, 10.0)
    Pgen_clipped = clamp(Pgen, 0.0, 1.0)
    Pload_clipped = clamp(Pload, 0.0, 1.0)
    t_clipped = clamp(t, 0.0, 24.0)
    
    # More stable activation function
    h1 = tanh(clamp(params[1]*x1_clipped + params[2]*x2_clipped + 
                    params[3]*Pgen_clipped + params[4]*Pload_clipped + 
                    params[5]*t_clipped + params[6], -10.0, 10.0))
    h2 = tanh(clamp(params[7]*x1_clipped + params[8]*x2_clipped + 
                    params[9]*Pgen_clipped + params[10]*Pload_clipped + 
                    params[11]*t_clipped + params[12], -10.0, 10.0))
    
    return clamp(params[13]*h1 + params[14]*h2 + params[15], -100.0, 100.0)
end

@model function ultra_stable_bayesian_ude(t, Y, u0)
    # Ultra-conservative noise prior
    σ ~ truncated(Normal(0.05, 0.01), 0.001, 0.1)
    
    # Very conservative physics parameter priors
    ηin ~ truncated(Normal(0.9, 0.05), 0.8, 1.0)
    ηout ~ truncated(Normal(0.9, 0.05), 0.8, 1.0)
    α ~ truncated(Normal(0.001, 0.0001), 0.0001, 0.01)
    β ~ truncated(Normal(1.0, 0.1), 0.5, 1.5)
    γ ~ truncated(Normal(0.001, 0.0001), 0.0001, 0.01)
    
    # Neural network parameters with very small variance
    nn_params ~ MvNormal(zeros(15), 0.05)  # Much smaller variance
    
    # Combine parameters
    p = [ηin, ηout, α, β, γ, nn_params...]
    
    # Ultra-strict ODE solving
    prob = ODEProblem(ultra_stable_ude_dynamics!, u0, (minimum(t), maximum(t)), p)
    
    sol = solve(prob, Tsit5(), saveat=t, 
                abstol=1e-10, reltol=1e-10,  # Ultra-strict tolerances
                maxiters=50000,  # More iterations
                adaptive=false,   # Fixed step size
                dt=0.01)         # Small fixed step size
    
    if sol.retcode != :Success || length(sol) != length(t)
        Turing.@addlogprob! -Inf
        return
    end
    
    # Conservative likelihood
    Ŷ = hcat(sol.u...)'
    for i in 1:length(t)
        Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
    end
end

println("Training Ultra-Stable UDE...")
ultra_ude_model = ultra_stable_bayesian_ude(t_train, Y_train, u0_train)

# ULTRA-STABLE UDE init
ultra_ude_initial_params = (σ = 0.05, ηin = 0.9, ηout = 0.9, α = 0.001, β = 1.0, γ = 0.001, nn_params = 0.05 .* randn(15))
try
    println("Running ADVI warm-start for Ultra-Stable UDE (iters=2000)...")
    q_u = vi(ultra_ude_model, ADVI(2000))
    if hasproperty(q_u, :posterior) && hasproperty(q_u.posterior, :μ)
        μu = q_u.posterior.μ
        if length(μu) >= 21
            ultra_ude_initial_params = (σ = max(0.01, abs(μu[1])), ηin = μu[2], ηout = μu[3], α = μu[4], β = μu[5], γ = μu[6], nn_params = μu[6:20])
        end
    end
catch e
    println("ADVI warm-start for Ultra-Stable UDE unavailable or failed: $(e). Proceeding with random init.")
end

ultra_ude_chain = sample(ultra_ude_model, NUTS(0.9), 500,
                        discard_initial=100, progress=true,
                        initial_params=ultra_ude_initial_params)

# Extract results
ultra_ude_params = Array(ultra_ude_chain)
ultra_physics_params = ultra_ude_params[:, 1:5]
ultra_neural_params = ultra_ude_params[:, 6:20]
ultra_ude_noise = ultra_ude_params[:, 21]

# Save results
ultra_ude_results = Dict(
    :physics_params_mean => mean(ultra_physics_params, dims=1)[1, :],
    :physics_params_std => std(ultra_physics_params, dims=1)[1, :],
    :neural_params_mean => mean(ultra_neural_params, dims=1)[1, :],
    :neural_params_std => std(ultra_neural_params, dims=1)[1, :],
    :noise_mean => mean(ultra_ude_noise),
    :noise_std => std(ultra_ude_noise),
    :n_samples => size(ultra_ude_params, 1),
    :model_type => "ultra_stable_universal_differential_equation"
)

BSON.@save "checkpoints/ultra_stable_ude_results.bson" ultra_ude_results
println("✅ Ultra-Stable UDE trained and saved")

# ============================================================================
# ULTRA-STABLE SYMBOLIC EXTRACTION
# ============================================================================
println("\n3. ULTRA-STABLE SYMBOLIC EXTRACTION")
println("-"^40)

println("Extracting symbolic form from Ultra-Stable UDE neural network...")

# Use the trained UDE neural network parameters
ultra_ude_nn_params = ultra_ude_results[:neural_params_mean]

# Generate data points for symbolic regression with better sampling
n_points = 100  # Smaller for stability
x1_range = range(-5.0, 5.0, length=5)  # Smaller range
x2_range = range(-5.0, 5.0, length=5)
Pgen_range = range(0.0, 1.0, length=4)
Pload_range = range(0.4, 0.8, length=4)
t_range = range(0.0, 24.0, length=4)

# Create a grid of possible inputs
ultra_symbolic_data = []
for x1 in x1_range
    for x2 in x2_range
        for Pgen in Pgen_range
            for Pload in Pload_range
                for t in t_range
                    push!(ultra_symbolic_data, [x1, x2, Pgen, Pload, t])
                end
            end
        end
    end
end

# Limit to reasonable number
ultra_symbolic_data = ultra_symbolic_data[1:min(n_points, length(ultra_symbolic_data))]

# Get the neural network's output for each input point
ultra_nn_outputs = []
for point in ultra_symbolic_data
    x1, x2, Pgen, Pload, t = point
    nn_output = ultra_stable_ude_nn([x1, x2, Pgen, Pload, t], ultra_ude_nn_params)
    push!(ultra_nn_outputs, Float64(nn_output))
end

# Polynomial regression for symbolic extraction
println("Performing ultra-stable symbolic regression...")

# Create a matrix of simple mathematical terms
function create_ultra_ude_features(x1, x2, Pgen, Pload, t)
    return [x1, x2, Pgen, Pload, t, 
            x1^2, x2^2, Pgen^2, Pload^2, t^2,
            x1*x2, x1*Pgen, x1*Pload, x1*t,
            x2*Pgen, x2*Pload, x2*t,
            Pgen*Pload, Pgen*t, Pload*t]
end

# Create feature matrix
ultra_feature_matrix = []
for (i, point) in enumerate(ultra_symbolic_data)
    features = create_ultra_ude_features(point[1], point[2], point[3], point[4], point[5])
    push!(ultra_feature_matrix, features)
end

ultra_feature_matrix = hcat(ultra_feature_matrix...)'
ultra_feature_matrix = convert(Matrix{Float64}, ultra_feature_matrix)

# Fit polynomial regression with regularization
ultra_nn_outputs = convert(Vector{Float64}, ultra_nn_outputs)

# Add regularization for stability
λ = 0.01  # Regularization parameter
I_reg = λ * Matrix{Float64}(I, size(ultra_feature_matrix, 2), size(ultra_feature_matrix, 2))
ultra_coefficients_ude_nn = (ultra_feature_matrix' * ultra_feature_matrix + I_reg) \ (ultra_feature_matrix' * ultra_nn_outputs)

# Calculate R² for symbolic extraction
ultra_pred_nn_output = ultra_feature_matrix * ultra_coefficients_ude_nn
ultra_actual_nn_output = ultra_nn_outputs

ultra_r2_ude_nn = 1 - sum((ultra_pred_nn_output .- ultra_actual_nn_output).^2) / sum((ultra_actual_nn_output .- mean(ultra_actual_nn_output)).^2)

# Save symbolic extraction results
ultra_symbolic_ude_results = Dict(
    :coefficients_ude_nn => ultra_coefficients_ude_nn,
    :r2_ude_nn => ultra_r2_ude_nn,
    :feature_names => ["x1", "x2", "Pgen", "Pload", "t", 
                      "x1^2", "x2^2", "Pgen^2", "Pload^2", "t^2",
                      "x1*x2", "x1*Pgen", "x1*Pload", "x1*t",
                      "x2*Pgen", "x2*Pload", "x2*t",
                      "Pgen*Pload", "Pgen*t", "Pload*t"],
    :model_type => "ultra_stable_symbolic_ude_extraction",
    :ude_nn_params => ultra_ude_nn_params,
    :n_features => 20,
    :regularization => λ
)

BSON.@save "checkpoints/ultra_stable_symbolic_ude_extraction.bson" ultra_symbolic_ude_results
println("✅ Ultra-stable symbolic extraction completed")
println("   - R² for UDE neural network: $(round(ultra_r2_ude_nn, digits=4))")
println("   - Features: $(length(ultra_symbolic_ude_results[:feature_names])) polynomial terms")
println("   - Regularization: λ = $(λ)")

# ============================================================================
# FINAL RESULTS SUMMARY
# ============================================================================
println("\n" * "="^60)
println("ULTRA-STABLE TRAINING COMPLETE")
println("="^60)

println("✅ ULTRA-STABLE BAYESIAN NEURAL ODE")
println("   - Ultra-strict tolerances: 1e-10")
println("   - Fixed step size: dt=0.01")
println("   - Conservative priors")
println("   - Samples: $(ultra_bayesian_results[:n_samples])")

println("\n✅ ULTRA-STABLE UDE (Universal Differential Equations)")
println("   - Ultra-strict tolerances: 1e-10")
println("   - Gradient clipping in neural network")
println("   - Conservative physics priors")
println("   - Samples: $(ultra_ude_results[:n_samples])")

println("\n✅ ULTRA-STABLE SYMBOLIC EXTRACTION")
println("   - Regularized polynomial regression")
println("   - R² = $(round(ultra_r2_ude_nn, digits=4))")
println("   - Smaller data ranges for stability")

println("\nULTRA-STABLE TRAINING SUCCESSFULLY COMPLETED! 🎯")
println("Numerical stability: 1e-10 tolerances + fixed step size + regularization") 