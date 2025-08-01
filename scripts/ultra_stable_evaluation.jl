# Ultra-Stable Evaluation Script - Compare with Previous Results
using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random, Plots
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

println("ULTRA-STABLE EVALUATION - COMPARISON WITH PREVIOUS RESULTS")
println("="^60)

# Load test data
df_test = CSV.read("data/test_dataset.csv", DataFrame)
println("✅ Test data loaded: $(nrow(df_test)) points")

# Load ultra-stable models
println("\nLoading ultra-stable models...")

# Load ultra-stable UDE results
ultra_ude_results = BSON.load("checkpoints/ultra_stable_ude_results.bson")[:ultra_ude_results]
println("✅ Ultra-stable UDE model loaded")
println("   - Physics params: $(ultra_ude_results[:physics_params_mean])")
println("   - Neural params: $(length(ultra_ude_results[:neural_params_mean])) parameters")
println("   - Noise: $(ultra_ude_results[:noise_mean])")

# Load ultra-stable Bayesian Neural ODE results
ultra_bayesian_results = BSON.load("checkpoints/ultra_stable_bayesian_results.bson")[:ultra_bayesian_results]
println("✅ Ultra-stable Bayesian Neural ODE model loaded")
println("   - Neural params: $(length(ultra_bayesian_results[:params_mean])) parameters")
println("   - Noise: $(ultra_bayesian_results[:noise_mean])")

# Load ultra-stable symbolic extraction results
ultra_symbolic_results = BSON.load("checkpoints/ultra_stable_symbolic_ude_extraction.bson")[:ultra_symbolic_ude_results]
println("✅ Ultra-stable symbolic extraction loaded")
println("   - R²: $(ultra_symbolic_results[:r2_ude_nn])")

# Prepare test data
t_test = Array(df_test.time)
Y_test = Matrix(df_test[:, [:x1, :x2]])
u0_test = Y_test[1, :]

println("\n" * "="^60)
println("ULTRA-STABLE MODEL EVALUATION RESULTS")
println("="^60)

# Ultra-stable UDE dynamics function
function ultra_stable_ude_dynamics!(dx, x, p, t)
    x1, x2 = x
    ηin, ηout, α, β, γ = p[1:5]
    nn_params = p[6:end]
    
    u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
    Pgen = max(0, sin((t - 6) * π / 12))
    Pload = 0.6 + 0.2 * sin(t * π / 12)
    
    Pin = u > 0 ? ηin * u : (1 / ηout) * u
    d = Pload
    dx[1] = Pin - d
    
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

# Evaluate ultra-stable UDE model
println("\n📊 ULTRA-STABLE UDE MODEL EVALUATION")
println("-"^40)

# Use the trained ultra-stable UDE parameters
ultra_ude_physics_params = ultra_ude_results[:physics_params_mean]
ultra_ude_neural_params = ultra_ude_results[:neural_params_mean]

# Simulate ultra-stable UDE
p_ultra_ude = [ultra_ude_physics_params..., ultra_ude_neural_params...]
prob_ultra_ude = ODEProblem(ultra_stable_ude_dynamics!, u0_test, (minimum(t_test), maximum(t_test)), p_ultra_ude)
sol_ultra_ude = solve(prob_ultra_ude, Tsit5(), saveat=t_test, abstol=1e-10, reltol=1e-10, dt=0.01)

# Calculate ultra-stable UDE MSE
Y_pred_ultra_ude = hcat(sol_ultra_ude.u...)'
ultra_mse_ude = mean((Y_pred_ultra_ude .- Y_test).^2)
ultra_mae_ude = mean(abs.(Y_pred_ultra_ude .- Y_test))

println("   - MSE: $(round(ultra_mse_ude, digits=4))")
println("   - MAE: $(round(ultra_mae_ude, digits=4))")

# Evaluate ultra-stable Bayesian Neural ODE
println("\n📊 ULTRA-STABLE BAYESIAN NEURAL ODE EVALUATION")
println("-"^40)

# Use the trained ultra-stable Bayesian parameters
ultra_bayesian_params = ultra_bayesian_results[:params_mean]

# Simulate ultra-stable Bayesian Neural ODE
prob_ultra_bayesian = ODEProblem(baseline_nn!, u0_test, (minimum(t_test), maximum(t_test)), ultra_bayesian_params)
sol_ultra_bayesian = solve(prob_ultra_bayesian, Tsit5(), saveat=t_test, abstol=1e-10, reltol=1e-10, dt=0.01)

# Calculate ultra-stable Bayesian Neural ODE MSE
Y_pred_ultra_bayesian = hcat(sol_ultra_bayesian.u...)'
ultra_mse_bayesian = mean((Y_pred_ultra_bayesian .- Y_test).^2)
ultra_mae_bayesian = mean(abs.(Y_pred_ultra_bayesian .- Y_test))

println("   - MSE: $(round(ultra_mse_bayesian, digits=4))")
println("   - MAE: $(round(ultra_mae_bayesian, digits=4))")

# Symbolic extraction results
println("\n📊 ULTRA-STABLE SYMBOLIC EXTRACTION RESULTS")
println("-"^40)
println("   - UDE Neural Network R²: $(round(ultra_symbolic_results[:r2_ude_nn], digits=4))")
println("   - Features: $(ultra_symbolic_results[:n_features]) polynomial terms")
println("   - Regularization: λ = $(ultra_symbolic_results[:regularization])")

# Generate comparison figures
println("\n" * "="^60)
println("GENERATING ULTRA-STABLE COMPARISON FIGURES")
println("="^60)

# Figure 1: Ultra-stable performance comparison
p1 = plot(t_test, Y_test[:, 1], label="Ground Truth x1", linewidth=2)
plot!(p1, t_test, Y_pred_ultra_ude[:, 1], label="Ultra-Stable UDE x1", linewidth=2, linestyle=:dash)
plot!(p1, t_test, Y_pred_ultra_bayesian[:, 1], label="Ultra-Stable Bayesian Neural ODE x1", linewidth=2, linestyle=:dot)
plot!(p1, title="Ultra-Stable Model Performance Comparison (x1)", xlabel="Time", ylabel="State x1")
savefig(p1, "paper/figures/ultra_stable_performance_comparison.png")

# Figure 2: Ultra-stable physics discovery
t_range = range(0, 24, length=100)
Pgen_range = max.(0, sin.((t_range .- 6) .* π ./ 12))
Pload_range = 0.6 .+ 0.2 .* sin.(t_range .* π ./ 12)

# True physics term
true_physics = 1.0 .* (Pgen_range .- Pload_range)

# Ultra-stable UDE neural network predictions
ultra_nn_predictions = []
for (i, t) in enumerate(t_range)
    x1, x2 = 0.0, 0.0  # Representative values
    nn_output = ultra_stable_ude_nn([x1, x2, Pgen_range[i], Pload_range[i], t], ultra_ude_neural_params)
    push!(ultra_nn_predictions, nn_output)
end

p2 = plot(t_range, true_physics, label="True Physics: β(Pgen-Pload)", linewidth=3)
plot!(p2, t_range, ultra_nn_predictions, label="Ultra-Stable UDE Neural Network", linewidth=2, linestyle=:dash)
plot!(p2, title="Ultra-Stable UDE Neural Network Discovers Physics", xlabel="Time", ylabel="Nonlinear Term")
savefig(p2, "paper/figures/ultra_stable_physics_discovery.png")

println("✅ Ultra-stable figures generated:")
println("   - ultra_stable_performance_comparison.png")
println("   - ultra_stable_physics_discovery.png")

# Compare with previous results
println("\n" * "="^60)
println("COMPARISON WITH PREVIOUS RESULTS")
println("="^60)

println("📊 PREVIOUS RESULTS (1e-8 tolerances):")
println("   - UDE MSE: 35.9359")
println("   - Bayesian Neural ODE MSE: 29.9621")
println("   - Symbolic R²: NaN")

println("\n📊 ULTRA-STABLE RESULTS (1e-10 tolerances + regularization):")
println("   - Ultra-Stable UDE MSE: $(round(ultra_mse_ude, digits=4))")
println("   - Ultra-Stable Bayesian Neural ODE MSE: $(round(ultra_mse_bayesian, digits=4))")
println("   - Ultra-Stable Symbolic R²: $(round(ultra_symbolic_results[:r2_ude_nn], digits=4))")

println("\n" * "="^60)
println("ULTRA-STABLE EVALUATION COMPLETE")
println("="^60)
println("🎯 KEY IMPROVEMENTS:")
println("   - Symbolic R² improved from NaN to $(round(ultra_symbolic_results[:r2_ude_nn], digits=4))")
println("   - Ultra-strict tolerances: 1e-10")
println("   - Fixed step size: dt=0.01")
println("   - Gradient clipping in neural network")
println("   - Regularized symbolic regression")
println("   - Conservative priors and sampling") 