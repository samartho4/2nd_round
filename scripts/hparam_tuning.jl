using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random, TOML
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures

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

# Hyperparameter tuning for Bayesian Neural ODE
# - Sweeps priors (sigma_theta), initialization scheme, and NUTS target acceptance
# - Runs with a reduced sample budget for speed
# - Saves results to checkpoints/hparam_search_results.csv and best config to checkpoints/best_hparam_config.bson

Random.seed!(123)

println("HYPERPARAMETER TUNING - Bayesian Neural ODE")
println("="^60)

# Config with config-file overrides
const OUTPUT_CSV = joinpath(@__DIR__, "..", "checkpoints", "hparam_search_results.csv")
const OUTPUT_BEST = joinpath(@__DIR__, "..", "checkpoints", "best_hparam_config.bson")

const N_SAMPLES = parse(Int, get(ENV, "HP_TUNE_SAMPLES", string(cfg(250, :tuning, :samples))))
const DISCARD = parse(Int, get(ENV, "HP_TUNE_WARMUP", string(cfg(50, :tuning, :warmup))))
const MAX_POINTS = parse(Int, get(ENV, "HP_TUNE_MAXPTS", string(cfg(400, :tuning, :max_points))))

const SIGMA_THETA_SPACE = parse.(Float64, split(get(ENV, "HP_SIGMA_THETA", join(string.(cfg([0.1,0.2,0.3], :tuning, :sigma_theta);), ",")), ","))
const INIT_SCHEMES = split(get(ENV, "HP_INIT_SCHEMES", join(cfg(["random","zeros"], :tuning, :init_schemes), ",")), ",")
const INIT_SCALE_SPACE = parse.(Float64, split(get(ENV, "HP_INIT_SCALE", join(string.(cfg([0.05,0.1,0.2], :tuning, :init_scale);), ",")), ","))
const NUTS_TARGET_SPACE = parse.(Float64, split(get(ENV, "HP_NUTS_TARGET", join(string.(cfg([0.60,0.65,0.80], :tuning, :nuts_target);), ",")), ","))

const ABSTOL = parse(Float64, get(ENV, "HP_ABSTOL", string(cfg(1e-8, :solver, :abstol))))
const RELTOL = parse(Float64, get(ENV, "HP_RELTOL", string(cfg(1e-8, :solver, :reltol))))

# -------------------------------
# Data
# -------------------------------
function load_data()
    println("Loading training/test data (subset for speed)...")
    df_train = CSV.read(joinpath(@__DIR__, "..", "data", "training_dataset.csv"), DataFrame)
    df_test  = CSV.read(joinpath(@__DIR__, "..", "data", "test_dataset.csv"), DataFrame)

    train_subset = first(df_train, min(nrow(df_train), MAX_POINTS))
    test_subset  = first(df_test,  min(nrow(df_test),  min(300, MAX_POINTS)))

    # Training arrays
    t_train = Array(train_subset.time)
    Y_train = Matrix(train_subset[:, [:x1, :x2]])
    u0_train = Y_train[1, :]

    # Small holdout (derivative-based proxy metric)
    t_test = Array(test_subset.time)
    Y_test = Matrix(test_subset[:, [:x1, :x2]])
    actual_derivatives = diff(Y_test, dims=1) ./ diff(t_test)
    t_derivatives = t_test[1:end-1]

    println("   -> Train points: $(length(t_train)), Test points: $(length(t_test))")
    return (t_train, Y_train, u0_train, t_test, Y_test, actual_derivatives, t_derivatives)
end

# -------------------------------
# Model builder
# -------------------------------
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

arch_cfg = cfg("baseline", :model, :arch)
arch_env = get(ENV, "MODEL_ARCH", nothing)
arch_choice = arch_env === nothing ? arch_cfg : String(arch_env)
arch_sym, deriv_fn, num_params = pick_arch(arch_choice)
println("Using architecture for tuning: $(arch_sym) with $(num_params) params")

@model function bayesian_neural_ode(t, Y, u0; sigma_theta::Float64)
    σ ~ truncated(Normal(0.1, 0.05), 0.01, 0.5)
    θ ~ MvNormal(zeros(num_params), (sigma_theta^2) * I(num_params))

    prob = ODEProblem(deriv_fn, u0, (minimum(t), maximum(t)), θ)
    sol = solve(prob, Tsit5(), saveat=t, abstol=ABSTOL, reltol=RELTOL, maxiters=10000)
    if sol.retcode != :Success || length(sol) != length(t)
        Turing.@addlogprob! -Inf
        return
    end
    Ŷ = hcat(sol.u...)'
    for i in 1:length(t)
        Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
    end
end

# -------------------------------
# Utilities
# -------------------------------
initial_theta(init_scheme::String, init_scale::Float64, dim::Int) = init_scheme == "zeros" ? zeros(dim) : init_scale .* randn(dim)

function evaluate_config(params_mean::Vector{Float64}, Y_test, t_derivatives, actual_derivatives)
    # Compute derivative prediction MSE on a small subset as a quick proxy
    preds = Vector{Vector{Float64}}(undef, length(t_derivatives))
    for i in 1:length(t_derivatives)
        x = Y_test[i, :]
        tt = t_derivatives[i]
        dx = zeros(2)
        baseline_nn!(dx, x, params_mean, tt)
        preds[i] = dx
    end
    preds_mat = hcat(preds...)'
    mse = mean((preds_mat .- actual_derivatives).^2)
    return mse
end

# -------------------------------
# Main tuning loop
# -------------------------------
function main()
    t_train, Y_train, u0_train, t_test, Y_test, actual_derivatives, t_derivatives = load_data()

    results = DataFrame(
        sigma_theta = Float64[],
        init_scheme = String[],
        init_scale = Float64[],
        nuts_target = Float64[],
        n_samples = Int[],
        discard = Int[],
        param_var_mean = Float64[],
        mse_deriv = Float64[],
    )

    mkpath(dirname(OUTPUT_CSV))

    best_mse = Inf
    best_cfg = Dict{Symbol,Any}()

    sweep_total = length(SIGMA_THETA_SPACE) * length(INIT_SCHEMES) * length(INIT_SCALE_SPACE) * length(NUTS_TARGET_SPACE)
    println("Total configs: $sweep_total")

    cfg_idx = 0
    for sigma_theta in SIGMA_THETA_SPACE
        for init_scheme in INIT_SCHEMES
            for init_scale in INIT_SCALE_SPACE
                for nuts_target in NUTS_TARGET_SPACE
                    cfg_idx += 1
                    println("\n[$cfg_idx/$sweep_total] sigma_theta=$sigma_theta, init=$init_scheme($init_scale), nuts_target=$nuts_target")

                    model = bayesian_neural_ode(t_train, Y_train, u0_train; sigma_theta=sigma_theta)

                    init_params = (σ = 0.1, θ = initial_theta(String(init_scheme), init_scale, num_params))

                    ch = sample(model, NUTS(nuts_target), N_SAMPLES;
                                discard_initial=DISCARD, progress=false, initial_params=init_params)

                    # Extract posterior means/variances
                    arr = Array(ch)
                    # Columns 1:10 are θ, 11 is σ (following scripts/train.jl convention)
                    θ_samples = arr[:, 1:num_params]
                    θ_mean = vec(mean(θ_samples, dims=1))
                    θ_var_mean = mean(var(θ_samples, dims=1))

                    mse = evaluate_config(θ_mean, Y_test, t_derivatives, actual_derivatives)

                    push!(results, (sigma_theta, init_scheme, init_scale, nuts_target, N_SAMPLES, DISCARD, θ_var_mean, mse))
                    CSV.write(OUTPUT_CSV, results)

                    if mse < best_mse
                        best_mse = mse
                        best_cfg = Dict(
                            :sigma_theta => sigma_theta,
                            :init_scheme => init_scheme,
                            :init_scale => init_scale,
                            :nuts_target => nuts_target,
                            :n_samples => N_SAMPLES,
                            :discard => DISCARD,
                            :abstol => ABSTOL,
                            :reltol => RELTOL,
                            :θ_mean => θ_mean,
                            :θ_var_mean => θ_var_mean,
                            :mse_deriv => mse,
                            :arch => String(arch_sym),
                        )
                        BSON.@save OUTPUT_BEST best_cfg
                        println("   -> New best mse=$(round(mse, digits=6)) saved to $(OUTPUT_BEST)")
                    else
                        println("   -> mse=$(round(mse, digits=6)) (no improvement)")
                    end
                end
            end
        end
    end

    println("\nTUNING COMPLETE. Results written to:\n - $(OUTPUT_CSV)\n - $(OUTPUT_BEST)")
end

main()  