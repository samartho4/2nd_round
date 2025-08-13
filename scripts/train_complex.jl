# Fixed Training Script - Actually implements the 3 objectives
using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random, TOML
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

Random.seed!(42)

println("FIXED TRAINING - IMPLEMENTING THE 3 OBJECTIVES")
println("="^60)

# Load data
df_train = CSV.read("data/training_dataset.csv", DataFrame)
df_test = CSV.read("data/test_dataset.csv", DataFrame)

# Load optional config
const CONFIG_PATH = joinpath(@__DIR__, "..", "config", "config.toml")
config = Dict{String,Any}()
if isfile(CONFIG_PATH)
    try
        config = TOML.parsefile(CONFIG_PATH)
        println("Loaded config from $(CONFIG_PATH)")
    catch e
        println("Warning: failed to parse config at $(CONFIG_PATH): $(e)")
    end
end
cfg(default, ks...) = begin
    v = config
    for k in ks
        if v isa Dict && haskey(v, String(k))
            v = v[String(k)]
        else
            return default
        end
    end
    return v
end
# Use larger subset for better training, but not too large to avoid ODE solver issues
subset_size = parse(Int, get(ENV, "TRAIN_SUBSET_SIZE", string(cfg(1500, :train, :subset_size))))
df_train_subset = df_train[1:subset_size, :]
df_test_subset = df_test[1:min(300, nrow(df_test)), :]

t_train = Array(df_train_subset.time)
Y_train = Matrix(df_train_subset[:, [:x1, :x2]])
u0_train = Y_train[1, :]

t_test = Array(df_test_subset.time)
Y_test = Matrix(df_test_subset[:, [:x1, :x2]])
u0_test = Y_test[1, :]

println("Data loaded: $(size(Y_train,1)) train, $(size(Y_test,1)) test points")

# Attempt to load best hyperparameter config (if exists)
const BEST_CFG_PATH = joinpath(@__DIR__, "..", "checkpoints", "best_hparam_config.bson")
# Allow training sample configuration via ENV
const TRAIN_SAMPLES = parse(Int, get(ENV, "TRAIN_SAMPLES", string(cfg(1000, :train, :samples))))
const TRAIN_WARMUP  = parse(Int, get(ENV, "TRAIN_WARMUP",  string(cfg(20, :train, :warmup))))
best_cfg = nothing
try
    if isfile(BEST_CFG_PATH)
        d = BSON.load(BEST_CFG_PATH)
        global best_cfg = d[:best_cfg]
        println("Loaded best hyperparameter config from: $(BEST_CFG_PATH)")
        println("   -> sigma_theta=$(best_cfg[:sigma_theta]), init=$(best_cfg[:init_scheme])($(best_cfg[:init_scale])), nuts_target=$(best_cfg[:nuts_target])")
    else
        println("No best hyperparameter config found; using defaults.")
    end
catch e
    println("Warning: failed to load best hyperparameter config: $(e)")
end

# Helper for initialization
function initial_theta(init_scheme::String, init_scale::Float64, dim::Int)
    if init_scheme == "zeros"
        return zeros(dim)
    else
        return init_scale .* randn(dim)
    end
end

# Choose architecture
function pick_arch(arch::AbstractString)
    arch = lowercase(String(arch))
    if arch == "baseline"; return (:baseline, baseline_nn!, 10)
    elseif arch == "baseline_bias"; return (:baseline_bias, baseline_nn_bias!, 14)
    elseif arch == "deep"; return (:deep, deep_nn!, 26)
    else
        println("Unknown arch=$(arch); defaulting to baseline")
        return (:baseline, baseline_nn!, 10)
    end
end

arch_env = get(ENV, "MODEL_ARCH", nothing)
arch_cfg = cfg("baseline", :model, :arch)
arch_choice = arch_env === nothing ? arch_cfg : String(arch_env)
arch_sym, deriv_fn, num_params = pick_arch(arch_choice)
println("Using architecture: $(arch_sym) with $(num_params) params")

# ============================================================================
# OBJECTIVE 1: Bayesian Neural ODE (Replace full ODE with neural network)
# ============================================================================
println("\n1. IMPLEMENTING BAYESIAN NEURAL ODE")
println("-"^40)

@model function bayesian_neural_ode(t, Y, u0)
    # Use tuned prior variance if available
    sigma_theta = isnothing(best_cfg) ? 0.3 : Float64(best_cfg[:sigma_theta])

    σ ~ truncated(Normal(0.1, 0.05), 0.01, 0.5)
    θ ~ MvNormal(zeros(num_params), (sigma_theta^2) * I(num_params))

    prob = ODEProblem(deriv_fn, u0, (minimum(t), maximum(t)), θ)
    sol = solve(prob, Tsit5(), saveat=t, abstol=1e-8, reltol=1e-8, maxiters=10000)

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

# Initialization: use tuned scheme if available
init_scheme = isnothing(best_cfg) ? "zeros" : String(best_cfg[:init_scheme])
init_scale  = isnothing(best_cfg) ? 0.1      : Float64(best_cfg[:init_scale])

initial_params = (σ = 0.1, θ = initial_theta(init_scheme, init_scale, num_params))

target_accept = isnothing(best_cfg) ? 0.65 : Float64(best_cfg[:nuts_target])

bayesian_chain = sample(bayesian_model, NUTS(target_accept), TRAIN_SAMPLES, discard_initial=TRAIN_WARMUP, progress=true, initial_params=initial_params)

# Extract results
bayesian_params = Array(bayesian_chain)[:, 1:num_params]
bayesian_noise = Array(bayesian_chain)[:, num_params+1]

# Save results properly
bayesian_results = Dict(
    :params_mean => mean(bayesian_params, dims=1)[1, :],
    :params_std => std(bayesian_params, dims=1)[1, :],
    :noise_mean => mean(bayesian_noise),
    :noise_std => std(bayesian_noise),
    :n_samples => size(bayesian_params, 1),
    :model_type => "bayesian_neural_ode",
    :arch => String(arch_sym),
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
    # Unpack known physics parameters and unknown nn_params
    x1, x2 = x
    ηin, ηout, α, β, γ = p[1:5]  # Physics parameters
    nn_params = p[6:end]          # Neural parameters (15)
    
    # Control and power inputs (same as original)
    u = t % 24 < 6 ? 1.0 : (t % 24 < 18 ? 0.0 : -0.8)
    Pgen = max(0, sin((t - 6) * π / 12))
    Pload = 0.6 + 0.2 * sin(t * π / 12)
    
    # Part 1: The physics we KNOW    Energy storage dynamics (physics only)
    Pin = u > 0 ? ηin * u : (1 / ηout) * u
    d = Pload
    dx[1] = Pin - d
    
    # Part 2: The physics we want to DISCOVER  Grid dynamics: physics + neural network for nonlinear term
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
    
    # Solve the ODE using our hybrid ude_dynamics! function  UDE solution with strict tolerances for numerical stability
    prob = ODEProblem(ude_dynamics!, u0, (minimum(t), maximum(t)), p)
    
    sol = solve(prob, Tsit5(), saveat=t, abstol=1e-8, reltol=1e-8, maxiters=10000)
    
    if sol.retcode != :Success || length(sol) != length(t)
        Turing.@addlogprob! -Inf
        return
    end
    # ... (Likelihood is the same) ...
    Ŷ = hcat(sol.u...)'
    for i in 1:length(t)
        Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
    end
end

# When we train this model, we are asking Turing to find the best values for both the physical parameters (like α) and NN weights at the same time. Train UDE
println("Training UDE...")
ude_model = bayesian_ude(t_train, Y_train, u0_train)

# Provide explicit initial parameters for better stability
ude_initial_params = (σ = 0.1, ηin = 0.9, ηout = 0.9, α = 0.001, β = 1.0, γ = 0.001, nn_params = zeros(15))

ude_chain = sample(ude_model, NUTS(0.65), 1000, discard_initial=20, progress=true, initial_params=ude_initial_params)

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

# // Get the best-fit parameters for the UDE's neural network
ude_nn_params = ude_results[:neural_params_mean]

# Generate data points for symbolic regression
n_points = 200
x1_range = range(-10.0, 10.0, length=8)
x2_range = range(-10.0, 10.0, length=8)
Pgen_range = range(0.0, 1.0, length=5)
Pload_range = range(0.4, 0.8, length=5)
t_range = range(0.0, 24.0, length=5)

# Create a grid of possible inputs (x1, x2, Pgen, Pload, t)
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

# Get the neural network's output for each input point
nn_outputs = []
for point in symbolic_data
    x1, x2, Pgen, Pload, t = point
    nn_output = simple_ude_nn([x1, x2, Pgen, Pload, t], ude_nn_params)
    push!(nn_outputs, Float64(nn_output))
end

# Polynomial regression for symbolic extraction of UDE neural network
println("Performing symbolic regression on UDE neural network...")

# Build polynomial feature matrix Φ and target y
feature_names = String[]
function build_features(x1, x2, Pgen, Pload, t)
    feats = [
        1.0,
        x1, x2, Pgen, Pload, t,
        x1^2, x2^2, Pgen^2, Pload^2, t^2,
        x1*Pgen, x2*Pgen, x1*Pload, x2*Pload, Pgen*Pload,
        x1*t, x2*t, Pgen*t, Pload*t
    ]
    return feats
end
feature_names = [
    "1", "x1","x2","Pgen","Pload","t",
    "x1^2","x2^2","Pgen^2","Pload^2","t^2",
    "x1*Pgen","x2*Pgen","x1*Pload","x2*Pload","Pgen*Pload",
    "x1*t","x2*t","Pgen*t","Pload*t"
]

n_points = min(2000, length(symbolic_data))
Φ = Array{Float64}(undef, n_points, length(feature_names))
y = Array{Float64}(undef, n_points)

for (i, point) in enumerate(symbolic_data[1:n_points])
    x1, x2, Pgen, Pload, t = point
    Φ[i, :] = build_features(x1, x2, Pgen, Pload, t)
    y[i] = simple_ude_nn([x1, x2, Pgen, Pload, t], neural_params) # Use neural_params directly
end

# Standardize features and target for numerical stability
μΦ = mean(Φ, dims=1)
σΦ = std(Φ, dims=1) .+ 1e-8
Φs = (Φ .- μΦ) ./ σΦ
μy = mean(y)
σy = std(y) + 1e-8
y_s = (y .- μy) ./ σy

# Ridge regression (closed form) on standardized system
λ = 1e-3
Ireg = Matrix{Float64}(I, size(Φs,2), size(Φs,2))
β_s = (Φs' * Φs .+ λ .* Ireg) \ (Φs' * y_s)

# Map back to original scale
# y ≈ (Φ - μΦ) / σΦ * β_s * σy + μy
β = (β_s ./ vec(σΦ)) .* σy
β0 = μy - sum((vec(μΦ) ./ vec(σΦ)) .* vec(β_s)) * σy

# Predictions and R² in original scale
ŷ = Φ * β .+ β0
ss_res = sum((y .- ŷ).^2)
ss_tot = sum((y .- mean(y)).^2)
R2 = 1 - ss_res / ss_tot

# Save symbolic extraction results for UDE neural network
symbolic_ude_results = Dict(
    :coeffs => β,
    :intercept => β0,
    :feature_names => feature_names,
    :R2 => R2,
    :n_points => n_points,
    :standardization => Dict(:mu => vec(μΦ), :sigma => vec(σΦ), :mu_y => μy, :sigma_y => σy),
    :model_type => "symbolic_ude_extraction",
)
BSON.@save "checkpoints/symbolic_ude_extraction.bson" symbolic_ude_results
println("✅ Symbolic extraction from UDE neural network completed")
println("   - R² for UDE neural network: $(R2)")
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
println("   - R² = $(round(R2, digits=4))")
println("   - Target: β * (Pgen - Pload) approximation")

println("\nALL 3 OBJECTIVES SUCCESSFULLY IMPLEMENTED! 🎯") 