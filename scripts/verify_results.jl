# Verification Script - Thoroughly Test Trained Models
using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random, Plots, TOML
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

# Config tolerances
const CONFIG_PATH = joinpath(@__DIR__, "..", "config", "config.toml")
config = isfile(CONFIG_PATH) ? TOML.parsefile(CONFIG_PATH) : Dict{String,Any}()
getcfg(dflt, ks...) = begin
    v = config
    for k in ks
        if v isa Dict && haskey(v, String(k))
            v = v[String(k)]
        else
            return dflt
        end
    end
    return v
end
a_tol = getcfg(1e-8, :solver, :abstol)
r_tol = getcfg(1e-8, :solver, :reltol)

println("THOROUGH VERIFICATION OF TRAINED MODELS")
println("="^60)

# Load test data
df_test = CSV.read("data/test_dataset.csv", DataFrame)
println("✅ Test data loaded: $(nrow(df_test)) points")

# Load train data and verify split integrity
println("\n🔍 VERIFICATION 0: TRAIN/TEST SPLIT INTEGRITY")
println("-"^40)
try
    df_train = CSV.read("data/training_dataset.csv", DataFrame)
    t_train_max = maximum(df_train.time)
    t_test_min = minimum(df_test.time)
    println("   Train max time = $(round(t_train_max, digits=3))")
    println("   Test  min time = $(round(t_test_min, digits=3))")
    if t_train_max <= 60.0 + 1e-6 && t_test_min >= 60.0 - 1e-6
        println("   ✅ Temporal split OK (≈0–60 train, 60+ test)")
    else
        println("   ❌ Temporal split mismatch: expected ~0–60 train and 60+ test")
    end
    if hasproperty(df_train, :scenario) && hasproperty(df_test, :scenario)
        inter = intersect(unique(df_train.scenario), unique(df_test.scenario))
        if isempty(inter)
            println("   ✅ Scenario sets disjoint between train and test")
        else
            println("   ❌ Scenario leakage detected between splits: $(inter)")
        end
    else
        println("   ℹ️  No scenario column found; skipping scenario leakage check")
    end
catch e
    println("   ⚠️  Could not verify train/test split: $(e)")
end

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
arch_name = haskey(bayesian_results, :arch) ? String(bayesian_results[:arch]) : "baseline"
function pick_arch(arch::AbstractString)
    a = lowercase(String(arch))
    if a == "baseline"; return (:baseline, baseline_nn!, 10)
    elseif a == "baseline_bias"; return (:baseline_bias, baseline_nn_bias!, 14)
    elseif a == "deep"; return (:deep, deep_nn!, 26)
    else; return (:baseline, baseline_nn!, 10)
    end
end
arch_sym, bayes_deriv_fn, _ = pick_arch(arch_name)
println("\nBayesian Neural ODE (arch=$(arch_sym)) Parameters (first 10):")
for i in 1:min(10, length(bayesian_params))
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
    nn_output = ude_nn_forward(x1, x2, Pgen, Pload, t, nn_params)
    dx[2] = -α * x2 + nn_output + γ * x1
end

# Test UDE simulation
try
    p_ude = [ude_physics..., ude_neural...]
    prob_ude = ODEProblem(ude_dynamics!, u0_test, (minimum(t_test), maximum(t_test)), p_ude)
    sol_ude = solve(prob_ude, Tsit5(), saveat=t_test, abstol=a_tol, reltol=r_tol)
    if string(sol_ude.retcode) == "Success"
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
    prob_bayesian = ODEProblem(bayes_deriv_fn, u0_test, (minimum(t_test), maximum(t_test)), bayesian_params)
    sol_bayesian = solve(prob_bayesian, Tsit5(), saveat=t_test, abstol=a_tol, reltol=r_tol)
    if string(sol_bayesian.retcode) == "Success"
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

test_inputs = [[1.0, 2.0, 0.5, 0.7, 12.0], [0.0, 0.0, 0.0, 0.6, 0.0], [-1.0, -2.0, 1.0, 0.8, 24.0]]

println("UDE Neural Network Outputs:")
for (i, input) in enumerate(test_inputs)
    output = ude_nn_forward(input[1], input[2], input[3], input[4], input[5], ude_neural)
    println("   Input $(i): $(round(output, digits=6))")
end

println("\n🔍 VERIFICATION 4: SYMBOLIC EXTRACTION UNIT TEST")
println("-"^40)
try
    sym = BSON.load("checkpoints/symbolic_ude_extraction.bson")[:symbolic_ude_results]
    βs = sym[:coeffs]; β0s = sym[:intercept]; names = sym[:feature_names]
    std = sym[:standardization]
    μΦ = collect(std[:mu]); σΦ = collect(std[:sigma]); μy = Float64(std[:mu_y]); σy = Float64(std[:sigma_y])
    β = (βs ./ σΦ) .* σy
    β0 = μy - sum((μΦ ./ σΦ) .* βs) * σy

    # Define feature builder consistent with symbolic extraction
    function build_features(x1, x2, Pgen, Pload, t)
        return [
            1.0,
            x1, x2, Pgen, Pload, t,
            x1^2, x2^2, Pgen^2, Pload^2, t^2,
            x1*Pgen, x2*Pgen, x1*Pload, x2*Pload, Pgen*Pload,
            x1*t, x2*t, Pgen*t, Pload*t
        ]
    end

    # Random subset
    n = min(200, nrow(df_test))
    idxs = sort(rand(1:nrow(df_test), n))
    y_nn = Float64[]
    y_poly = Float64[]
    for i in idxs
        t = df_test.time[i]; x1 = df_test.x1[i]; x2 = df_test.x2[i]
        Pgen = max(0, sin((t - 6) * π / 12)); Pload = 0.6 + 0.2 * sin(t * π / 12)
        push!(y_nn, ude_nn_forward(x1, x2, Pgen, Pload, t, ude_neural))
        Φ = build_features(x1, x2, Pgen, Pload, t)
        push!(y_poly, sum(β .* Φ) + β0)
    end
    # Compute R2
    ss_res = sum((y_nn .- y_poly).^2)
    ss_tot = sum((y_nn .- mean(y_nn)).^2)
    r2 = 1 - ss_res / ss_tot
    println("   R2(UDE NN vs polynomial) = $(round(r2, digits=6))")
    println("   Pgen coeff ≈ $(round(β[names .== "Pgen"][1], digits=4)), Pload coeff ≈ $(round(β[names .== "Pload"][1], digits=4))")
catch e
    println("   ⚠️  Symbolic unit test failed: $(e)")
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