# Fresh Evaluation Script - Using Newly Trained Models
using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random, Plots
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

println("FRESH EVALUATION - USING NEWLY TRAINED MODELS")
println("="^60)

# Load fresh test data
df_test = CSV.read("data/test_dataset.csv", DataFrame)
println("✅ Test data loaded: $(nrow(df_test)) points")

# Load newly trained models
println("\nLoading newly trained models...")

# Load UDE results
ude_results = BSON.load("checkpoints/ude_results_fixed.bson")[:ude_results]
println("✅ UDE model loaded")
println("   - Physics params: $(ude_results[:physics_params_mean])")
println("   - Neural params: $(length(ude_results[:neural_params_mean])) parameters")
println("   - Noise: $(ude_results[:noise_mean])")

# Load Bayesian Neural ODE results
bayesian_results = BSON.load("checkpoints/bayesian_neural_ode_results.bson")[:bayesian_results]
println("✅ Bayesian Neural ODE model loaded")
println("   - Neural params: $(length(bayesian_results[:params_mean])) parameters")
println("   - Noise: $(bayesian_results[:noise_mean])")

# Load symbolic extraction results
symbolic_results = BSON.load("checkpoints/symbolic_ude_extraction.bson")[:symbolic_ude_results]
println("✅ Symbolic extraction loaded")
println("   - R²: $(symbolic_results[:r2_ude_nn])")

# Prepare test data
t_test = Array(df_test.time)
Y_test = Matrix(df_test[:, [:x1, :x2]])
u0_test = Y_test[1, :]

println("\n" * "="^60)
println("FRESH MODEL EVALUATION RESULTS")
println("="^60)

# Evaluate UDE model
println("\n📊 UDE MODEL EVALUATION")
println("-"^40)

# Use the trained UDE parameters
ude_physics_params = ude_results[:physics_params_mean]
ude_neural_params = ude_results[:neural_params_mean]

# UDE dynamics function
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

# Simulate UDE
p_ude = [ude_physics_params..., ude_neural_params...]
prob_ude = ODEProblem(ude_dynamics!, u0_test, (minimum(t_test), maximum(t_test)), p_ude)
sol_ude = solve(prob_ude, Tsit5(), saveat=t_test, abstol=1e-8, reltol=1e-8)

# Calculate UDE MSE
Y_pred_ude = hcat(sol_ude.u...)'
mse_ude = mean((Y_pred_ude .- Y_test).^2)
mae_ude = mean(abs.(Y_pred_ude .- Y_test))

println("   - MSE: $(round(mse_ude, digits=4))")
println("   - MAE: $(round(mae_ude, digits=4))")

# Evaluate Bayesian Neural ODE
println("\n📊 BAYESIAN NEURAL ODE EVALUATION")
println("-"^40)

# Use the trained Bayesian parameters
bayesian_params = bayesian_results[:params_mean]

# Simulate Bayesian Neural ODE
prob_bayesian = ODEProblem(baseline_nn!, u0_test, (minimum(t_test), maximum(t_test)), bayesian_params)
sol_bayesian = solve(prob_bayesian, Tsit5(), saveat=t_test, abstol=1e-8, reltol=1e-8)

# Calculate Bayesian Neural ODE MSE
Y_pred_bayesian = hcat(sol_bayesian.u...)'
mse_bayesian = mean((Y_pred_bayesian .- Y_test).^2)
mae_bayesian = mean(abs.(Y_pred_bayesian .- Y_test))

println("   - MSE: $(round(mse_bayesian, digits=4))")
println("   - MAE: $(round(mae_bayesian, digits=4))")

# Symbolic extraction results
println("\n📊 SYMBOLIC EXTRACTION RESULTS")
println("-"^40)
println("   - UDE Neural Network R²: $(round(symbolic_results[:r2_ude_nn], digits=4))")
println("   - Features: $(symbolic_results[:n_features]) polynomial terms")

# Generate fresh figures
println("\n" * "="^60)
println("GENERATING FRESH FIGURES")
println("="^60)

# Figure 1: Performance Comparison
p1 = plot(t_test, Y_test[:, 1], label="Ground Truth x1", linewidth=2)
plot!(p1, t_test, Y_pred_ude[:, 1], label="UDE x1", linewidth=2, linestyle=:dash)
plot!(p1, t_test, Y_pred_bayesian[:, 1], label="Bayesian Neural ODE x1", linewidth=2, linestyle=:dot)
plot!(p1, title="Model Performance Comparison (x1)", xlabel="Time", ylabel="State x1")
savefig(p1, "paper/figures/fresh_performance_comparison.png")

# Figure 2: Physics Discovery
# Generate data for physics discovery plot
t_range = range(0, 24, length=100)
Pgen_range = max.(0, sin.((t_range .- 6) .* π ./ 12))
Pload_range = 0.6 .+ 0.2 .* sin.(t_range .* π ./ 12)

# True physics term
true_physics = 1.0 .* (Pgen_range .- Pload_range)

# UDE neural network predictions
nn_predictions = []
for (i, t) in enumerate(t_range)
    x1, x2 = 0.0, 0.0  # Representative values
    nn_output = simple_ude_nn([x1, x2, Pgen_range[i], Pload_range[i], t], ude_neural_params)
    push!(nn_predictions, nn_output)
end

p2 = plot(t_range, true_physics, label="True Physics: β(Pgen-Pload)", linewidth=3)
plot!(p2, t_range, nn_predictions, label="UDE Neural Network", linewidth=2, linestyle=:dash)
plot!(p2, title="UDE Neural Network Discovers Physics", xlabel="Time", ylabel="Nonlinear Term")
savefig(p2, "paper/figures/fresh_physics_discovery.png")

println("✅ Fresh figures generated:")
println("   - fresh_performance_comparison.png")
println("   - fresh_physics_discovery.png")

println("\n" * "="^60)
println("FRESH EVALUATION COMPLETE")
println("="^60)
println("📊 FRESH RESULTS SUMMARY:")
println("   - UDE MSE: $(round(mse_ude, digits=4))")
println("   - Bayesian Neural ODE MSE: $(round(mse_bayesian, digits=4))")
println("   - Symbolic R²: $(round(symbolic_results[:r2_ude_nn], digits=4))")
println("   - Training samples: $(ude_results[:n_samples])")
println("   - Numerical stability: 1e-8 tolerances") 