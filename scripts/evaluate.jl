# Dynamic Model Evaluation Script
using BSON, CSV, DataFrames, Statistics, DifferentialEquations
include("../src/microgrid_system.jl")
include("../src/neural_ode_architectures.jl")
using .NeuralNODEArchitectures

println("DYNAMIC MODEL EVALUATION")
println("="^50)

# Helper: pick architecture by name → (symbol, fn, num_params)
function pick_arch(arch::AbstractString)
    a = lowercase(String(arch))
    if a == "baseline"
        return (:baseline, baseline_nn!, 10)
    elseif a == "baseline_bias"
        return (:baseline_bias, baseline_nn_bias!, 14)
    elseif a == "deep"
        return (:deep, deep_nn!, 26)
    else
        println("Warning: unknown arch=$(arch); defaulting to baseline")
        return (:baseline, baseline_nn!, 10)
    end
end

# Load test dataset
println("Loading test dataset...")
df_test = CSV.read("data/test_dataset.csv", DataFrame)
println("✅ Test data loaded: $(nrow(df_test)) points")
println("   Columns: $(names(df_test))")

# Load Bayesian Neural ODE model
println("\nLoading Bayesian Neural ODE model...")
let
    try
        bayesian_file = BSON.load("checkpoints/bayesian_neural_ode_results.bson")
        global bayesian_results = bayesian_file[:bayesian_results]
    catch e
        error("Failed to load Bayesian results at checkpoints/bayesian_neural_ode_results.bson: $(e)")
    end
end

arch_name = haskey(bayesian_results, :arch) ? String(bayesian_results[:arch]) : "baseline"
arch_sym, bayes_deriv_fn, num_params = pick_arch(arch_name)
println("✅ Bayesian Neural ODE model loaded")
println("   - Architecture: $(arch_sym) ($(num_params) params)")
println("   - Model type: $(get(bayesian_results, :model_type, "unknown"))")
println("   - Parameters: $(length(bayesian_results[:params_mean]))")

# Load UDE model
println("\nLoading UDE model...")
let
    try
        ude_file = BSON.load("checkpoints/ude_results_fixed.bson")
        global ude_results = ude_file[:ude_results]
    catch e
        error("Failed to load UDE results at checkpoints/ude_results_fixed.bson: $(e)")
    end
end
println("✅ UDE model loaded")
println("   - Model type: $(get(ude_results, :model_type, "unknown"))")
println("   - Physics parameters: $(length(ude_results[:physics_params_mean]))")
println("   - Neural parameters: $(length(ude_results[:neural_params_mean]))")

# Prepare test data for evaluation
println("\nPreparing test data for evaluation...")
t_test = Array(df_test.time)
Y_test = Matrix(df_test[:, [:x1, :x2]])

# Calculate actual derivatives from test data
println("Computing actual derivatives...")
actual_derivatives = diff(Y_test, dims=1) ./ diff(t_test)
t_derivatives = t_test[1:end-1]  # Time points for derivatives

println("✅ Derivatives computed: $(size(actual_derivatives, 1)) points")

# Bayesian Neural ODE predictions and evaluation
println("\n" * "="^50)
println("BAYESIAN NEURAL ODE EVALUATION")
println("="^50)

bayesian_params = bayesian_results[:params_mean]
bayesian_predictions = []

for i in 1:length(t_derivatives)
    x = Y_test[i, :]
    t = t_derivatives[i]
    dx = zeros(2)
    bayes_deriv_fn(dx, x, bayesian_params, t)
    push!(bayesian_predictions, dx)
end

bayesian_predictions = hcat(bayesian_predictions...)'

# Calculate Bayesian Neural ODE metrics
bayesian_mse = mean((bayesian_predictions .- actual_derivatives).^2)
bayesian_mae = mean(abs.(bayesian_predictions .- actual_derivatives))
bayesian_r2 = 1 - sum((bayesian_predictions .- actual_derivatives).^2) / sum((actual_derivatives .- mean(actual_derivatives, dims=1)).^2)

println("Performance Metrics:")
println("   - MSE: $(round(bayesian_mse, digits=4))")
println("   - MAE: $(round(bayesian_mae, digits=4))")
println("   - R2: $(round(bayesian_r2, digits=4))")

# UDE predictions and evaluation
println("\n" * "="^50)
println("UDE EVALUATION")
println("="^50)

physics_params = ude_results[:physics_params_mean]
neural_params = ude_results[:neural_params_mean]

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
    
    # Neural network for nonlinear term
    h1 = tanh(nn_params[1]*x1 + nn_params[2]*x2 + nn_params[3]*Pgen + nn_params[4]*Pload + nn_params[5]*t + nn_params[6])
    h2 = tanh(nn_params[7]*x1 + nn_params[8]*x2 + nn_params[9]*Pgen + nn_params[10]*Pload + nn_params[11]*t + nn_params[12])
    nn_output = nn_params[13]*h1 + nn_params[14]*h2 + nn_params[15]
    
    dx[2] = -α * x2 + nn_output + γ * x1
end

# Make UDE predictions
ude_predictions = []
p_ude = [physics_params..., neural_params...]

for i in 1:length(t_derivatives)
    x = Y_test[i, :]
    t = t_derivatives[i]
    
    dx = zeros(2)
    ude_dynamics!(dx, x, p_ude, t)
    push!(ude_predictions, dx)
end

ude_predictions = hcat(ude_predictions...)'

# Calculate UDE metrics
ude_mse = mean((ude_predictions .- actual_derivatives).^2)
ude_mae = mean(abs.(ude_predictions .- actual_derivatives))
ude_r2 = 1 - sum((ude_predictions .- actual_derivatives).^2) / sum((actual_derivatives .- mean(actual_derivatives, dims=1)).^2)

println("Performance Metrics:")
println("   - MSE: $(round(ude_mse, digits=4))")
println("   - MAE: $(round(ude_mae, digits=4))")
println("   - R2: $(round(ude_r2, digits=4))")

# Load symbolic extraction results for comparison
println("\n" * "="^50)
println("SYMBOLIC EXTRACTION RESULTS")
println("="^50)

# Load original symbolic extraction
symbolic_file = BSON.load("checkpoints/symbolic_models_fixed.bson")
symbolic_results = symbolic_file[:symbolic_results]

println("Original Symbolic Extraction:")
println("   - R2 for dx1: $(round(symbolic_results[:r2_dx1], digits=4))")
println("   - R2 for dx2: $(round(symbolic_results[:r2_dx2], digits=4))")
println("   - Average R2: $(round(symbolic_results[:avg_r2], digits=4))")

# Load new UDE symbolic extraction
println("\n" * "="^50)
println("UDE NEURAL NETWORK SYMBOLIC EXTRACTION")
println("="^50)

symbolic_ude_file = BSON.load("checkpoints/symbolic_ude_extraction.bson")
symbolic_ude_results = symbolic_ude_file[:symbolic_ude_results]

println("UDE Neural Network Symbolic Extraction:")
println("   - R2 for UDE neural network: $(round(symbolic_ude_results[:R2], digits=4))")
println("   - Features: $(length(symbolic_ude_results[:feature_names])) polynomial terms")
println("   - Target: β * (Pgen - Pload) approximation")

# After loading models and preparing data

# Posterior Predictive: quick coverage check on first scenario window
try
    println("\nPOSTERIOR PREDICTIVE CHECKS (coverage on a small window)")
    # Identify a contiguous block in test set (first scenario)
    first_scn = df_test.scenario[1]
    block = filter(row -> row.scenario == first_scn, df_test)
    t_blk = Array(block.time)
    Y_blk = Matrix(block[:, [:x1, :x2]])
    x0_blk = Y_blk[1, :]

    # Helper to get samples if present
    bayes_param_samples = get(bayesian_results, :param_samples, nothing)
    ude_phys_samples = get(ude_results, :physics_samples, nothing)
    ude_nn_samples = get(ude_results, :neural_samples, nothing)

    # Bayesian ODE bands
    if bayes_param_samples !== nothing
        preds = Array{Float64}(undef, length(t_blk), 2, size(bayes_param_samples, 1))
        for (k, θ) in enumerate(eachrow(bayes_param_samples))
            prob = ODEProblem(bayes_deriv_fn, x0_blk, (t_blk[1], t_blk[end]), collect(θ))
            sol = solve(prob, Tsit5(), saveat=t_blk, abstol=1e-8, reltol=1e-8, maxiters=10000)
            preds[:, :, k] = hcat(sol.u...)'
        end
        med = mapslices(x -> median(x), preds; dims=3)[:, :, 1]
        lo = mapslices(x -> quantile(x, 0.05), preds; dims=3)[:, :, 1]
        hi = mapslices(x -> quantile(x, 0.95), preds; dims=3)[:, :, 1]
        cover = mean((Y_blk .>= lo) .& (Y_blk .<= hi))
        println("   Bayesian ODE 5–95% coverage: $(round(cover, digits=3))")
    end

    # UDE bands
    if ude_phys_samples !== nothing && ude_nn_samples !== nothing
        ns = min(size(ude_phys_samples, 1), size(ude_nn_samples, 1))
        preds = Array{Float64}(undef, length(t_blk), 2, ns)
        function ude_dyn_s!(dx, x, p, t)
            x1, x2 = x
            ηin, ηout, α, β, γ = p[1:5]
            nn_params = p[6:end]
            u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
            Pgen = max(0, sin((t - 6) * π / 12))
            Pload = 0.6 + 0.2 * sin(t * π / 12)
            Pin = u > 0 ? ηin * u : (1 / ηout) * u
            dx[1] = Pin - Pload
            nn_output = ude_nn_forward(x1, x2, Pgen, Pload, t, nn_params)
            dx[2] = -α * x2 + nn_output + γ * x1
        end
        for k in 1:ns
            p = [ude_phys_samples[k, :]..., ude_nn_samples[k, :]...]
            prob = ODEProblem(ude_dyn_s!, x0_blk, (t_blk[1], t_blk[end]), p)
            sol = solve(prob, Tsit5(), saveat=t_blk, abstol=1e-8, reltol=1e-8, maxiters=10000)
            preds[:, :, k] = hcat(sol.u...)'
        end
        med = mapslices(x -> median(x), preds; dims=3)[:, :, 1]
        lo = mapslices(x -> quantile(x, 0.05), preds; dims=3)[:, :, 1]
        hi = mapslices(x -> quantile(x, 0.95), preds; dims=3)[:, :, 1]
        cover = mean((Y_blk .>= lo) .& (Y_blk .<= hi))
        println("   UDE 5–95% coverage: $(round(cover, digits=3))")
    end
catch e
    println("   (PPC skipped): $(e)")
end

# Final comprehensive summary
println("\n" * "="^60)
println("COMPREHENSIVE PERFORMANCE SUMMARY")
println("="^60)

println("📊 DYNAMICALLY COMPUTED METRICS")
println("   Test dataset: $(nrow(df_test)) points")
println("   Evaluation points: $(size(actual_derivatives, 1))")

println("\n🏆 MODEL PERFORMANCE COMPARISON")
println("   Bayesian Neural ODE:")
println("     - MSE: $(round(bayesian_mse, digits=4))")
println("     - MAE: $(round(bayesian_mae, digits=4))")
println("     - R2: $(round(bayesian_r2, digits=4))")

println("\n   UDE (Universal Differential Equations):")
println("     - MSE: $(round(ude_mse, digits=4))")
println("     - MAE: $(round(ude_mae, digits=4))")
println("     - R2: $(round(ude_r2, digits=4))")

println("\n   Symbolic Extraction:")
println("     - Average R2: $(round(symbolic_results[:avg_r2], digits=4))")

println("\n   UDE Neural Network Symbolic Extraction:")
println("     - R2: $(round(symbolic_ude_results[:R2], digits=4))")
println("     - Target: β * (Pgen - Pload) approximation")

println("\n✅ EVALUATION COMPLETE")
println("All metrics computed dynamically from loaded models and test data.") 