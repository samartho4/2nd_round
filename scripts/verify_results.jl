# Verification Script - Thoroughly Test Trained Models
using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random, Plots
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

println("THOROUGH VERIFICATION OF TRAINED MODELS")
println("="^60)

# Load test data
df_test = CSV.read("data/test_dataset.csv", DataFrame)
println("✅ Test data loaded: $(nrow(df_test)) points")

# Load trained models
ude_results = BSON.load("checkpoints/ude_results_fixed.bson")[:ude_results]
bayesian_results = BSON.load("checkpoints/bayesian_neural_ode_results.bson")[:bayesian_results]

println("\n🔍 VERIFICATION 1: PARAMETER ANALYSIS")
println("-"^40)

# Check UDE parameters
ude_physics = ude_results[:physics_params_mean]
ude_neural = ude_results[:neural_params_mean]
println("UDE Physics Parameters:")
println("   ηin: $(ude_physics[1])")
println("   ηout: $(ude_physics[2])")
println("   α: $(ude_physics[3])")
println("   β: $(ude_physics[4])")
println("   γ: $(ude_physics[5])")

println("\nUDE Neural Parameters (first 10):")
for i in 1:10
    println("   θ$(i): $(ude_neural[i])")
end

# Check Bayesian parameters
bayesian_params = bayesian_results[:params_mean]
println("\nBayesian Neural Parameters (first 10):")
for i in 1:10
    println("   θ$(i): $(bayesian_params[i])")
end

println("\n🔍 VERIFICATION 2: MODEL SIMULATION TEST")
println("-"^40)

# Test on a small subset
subset_size = 100
t_test = Array(df_test.time[1:subset_size])
Y_test = Matrix(df_test[1:subset_size, [:x1, :x2]])
u0_test = Y_test[1, :]

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

# Test UDE simulation
try
    p_ude = [ude_physics..., ude_neural...]
    prob_ude = ODEProblem(ude_dynamics!, u0_test, (minimum(t_test), maximum(t_test)), p_ude)
    sol_ude = solve(prob_ude, Tsit5(), saveat=t_test, abstol=1e-8, reltol=1e-8)
    
    if sol_ude.retcode == :Success
        Y_pred_ude = hcat(sol_ude.u...)'
        mse_ude = mean((Y_pred_ude .- Y_test).^2)
        println("✅ UDE simulation successful")
        println("   - MSE: $(round(mse_ude, digits=4))")
        println("   - Solution length: $(length(sol_ude))")
    else
        println("❌ UDE simulation failed: $(sol_ude.retcode)")
    end
catch e
    println("❌ UDE simulation error: $(e)")
end

# Test Bayesian Neural ODE simulation
try
    prob_bayesian = ODEProblem(baseline_nn!, u0_test, (minimum(t_test), maximum(t_test)), bayesian_params)
    sol_bayesian = solve(prob_bayesian, Tsit5(), saveat=t_test, abstol=1e-8, reltol=1e-8)
    
    if sol_bayesian.retcode == :Success
        Y_pred_bayesian = hcat(sol_bayesian.u...)'
        mse_bayesian = mean((Y_pred_bayesian .- Y_test).^2)
        println("✅ Bayesian Neural ODE simulation successful")
        println("   - MSE: $(round(mse_bayesian, digits=4))")
        println("   - Solution length: $(length(sol_bayesian))")
    else
        println("❌ Bayesian Neural ODE simulation failed: $(sol_bayesian.retcode)")
    end
catch e
    println("❌ Bayesian Neural ODE simulation error: $(e)")
end

println("\n🔍 VERIFICATION 3: NEURAL NETWORK ACTIVATION TEST")
println("-"^40)

# Test if neural networks are actually doing anything
test_inputs = [[1.0, 2.0, 0.5, 0.7, 12.0], [0.0, 0.0, 0.0, 0.6, 0.0], [-1.0, -2.0, 1.0, 0.8, 24.0]]

println("UDE Neural Network Outputs:")
for (i, input) in enumerate(test_inputs)
    output = simple_ude_nn(input, ude_neural)
    println("   Input $(i): $(round(output, digits=6))")
end

println("\n🔍 VERIFICATION 4: PARAMETER SENSITIVITY TEST")
println("-"^40)

# Test if changing parameters actually changes the output
perturbed_params = copy(ude_neural)
perturbed_params[1] += 0.1

println("UDE Neural Network with original params:")
for (i, input) in enumerate(test_inputs)
    output = simple_ude_nn(input, ude_neural)
    println("   Input $(i): $(round(output, digits=6))")
end

println("\nUDE Neural Network with perturbed params:")
for (i, input) in enumerate(test_inputs)
    output = simple_ude_nn(input, perturbed_params)
    println("   Input $(i): $(round(output, digits=6))")
end

println("\n" * "="^60)
println("VERIFICATION SUMMARY")
println("="^60)

# Determine if models actually learned
ude_neural_variance = var(ude_neural)
bayesian_variance = var(bayesian_params)

println("📊 PARAMETER VARIANCE ANALYSIS:")
println("   - UDE neural parameters variance: $(round(ude_neural_variance, digits=8))")
println("   - Bayesian neural parameters variance: $(round(bayesian_variance, digits=8))")

if ude_neural_variance < 1e-6
    println("❌ UDE neural network did NOT learn meaningful parameters")
else
    println("✅ UDE neural network learned meaningful parameters")
end

if bayesian_variance < 1e-6
    println("❌ Bayesian Neural ODE did NOT learn meaningful parameters")
else
    println("✅ Bayesian Neural ODE learned meaningful parameters")
end

println("\n🎯 CONCLUSION:")
if ude_neural_variance < 1e-6 && bayesian_variance < 1e-6
    println("❌ BOTH MODELS FAILED TO LEARN - Training converged to trivial solutions")
    println("   This suggests the numerical stability fixes weren't sufficient")
    println("   The models need additional training improvements")
else
    println("✅ Models learned meaningful parameters")
end 