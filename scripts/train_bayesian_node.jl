# Bayesian Neural ODE Training Script
# ----------------------------------
# Objective 1: Replace full ODE with Bayesian Neural ODE and perform prediction / forecasting.
#
# Uses Turing.jl to place priors on the 10-parameter neural derivative (baseline_nn!)
# and an observation noise term.  Samples posterior with NUTS, then produces forecasts
# for train/val/test splits and writes metrics + figures.
#
# NOTE: This is a *thin* wrapper around the architecture defined in
#       src/NeuralNODEArchitectures.jl  (baseline_nn!).
#       For computational reasons, it subsamples the large dataset during demo runs.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using DifferentialEquations, Turing, CSV, DataFrames, BSON, Statistics, Random, Dates
include(joinpath(@__DIR__, "..", "src", "Microgrid.jl"))
include(joinpath(@__DIR__, "..", "src", "NeuralNODEArchitectures.jl"))
using .NeuralNODEArchitectures

Random.seed!(1234)

# ----------------------- CLI configuration -------------------------
n_samples = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100    # posterior samples after burn-in
max_points = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300    # per split subsampling for speed

println("Using NUTS samples=$n_samples, max_points per split=$max_points")

# ------------------------- Load dataset ---------------------------
function load_split(split::Symbol; max_points::Int)
    df = CSV.read(joinpath(@__DIR__, "..", "data", "$(split)_improved.csv"), DataFrame)
    if nrow(df) > max_points
        df = df[randperm(nrow(df))[1:max_points], :]
    end
    sort!(df, :time)
    t = Array(df.time)
    Y = Matrix(df[:, [:x1, :x2]])
    u0 = Y[1, :]
    return t, Y, u0
end

t_train, Y_train, u0      = load_split(:train; max_points=max_points)
t_val,   Y_val,   u0_val  = load_split(:val;  max_points=Int(round(0.5*max_points)))
t_test,  Y_test,  u0_test = load_split(:test; max_points=Int(round(0.5*max_points)))

println("Loaded train / val / test subsets for Bayesian NODE")

# ------------------------- Turing model ---------------------------
@model function bayesian_node(t, Y, u0)
    σ ~ truncated(Normal(0.05, 0.05), 1e-3, 0.5)
    θ ~ MvNormal(zeros(10), 0.3)

    prob = ODEProblem(baseline_nn!, u0, (minimum(t), maximum(t)), θ)
    sol  = solve(prob, Tsit5(), saveat=t, abstol=1e-5, reltol=1e-5, maxiters=5_000)

    if sol.retcode != :Success || length(sol) != length(t)
        Turing.@addlogprob! -Inf
        return
    end

    Ŷ = hcat(sol.u...)'
    for i in 1:length(t)
        Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
    end
end

println("Sampling posterior with NUTS (short demo chain)…")
model = bayesian_node(t_train, Y_train, u0)
chain = sample(model, NUTS(0.65), n_samples + 20; discard_initial=20, thinning=1, progress=true)

println("Posterior sampling done: $(length(chain)) samples")

# ------------------------- Forecast & Metrics ---------------------
function forecast(θ, u0, tspan; saveat_times)
    prob = ODEProblem(baseline_nn!, u0, tspan, θ)
    sol  = solve(prob, Tsit5(), saveat=saveat_times, abstol=1e-5, reltol=1e-5)
    return hcat(sol.u...)'
end

posterior_samples = Array(chain)[:, 1:10]  # columns for θ[1],θ[2],…

function metrics(Ŷ, Y)
    mse = mean((Ŷ .- Y).^2)
    mae = mean(abs.(Ŷ .- Y))
    r2  = 1 - sum((Ŷ .- Y).^2) / sum((Y .- mean(Y)).^2)
    return mse, mae, r2
end

function evaluate_split(t, Y, u0; nsamples=50)
    ns = min(nsamples, size(posterior_samples,1))
    idxs = rand(1:size(posterior_samples,1), ns)
    pred_list = [forecast(posterior_samples[i, :], u0, (minimum(t), maximum(t)); saveat_times=t) for i in idxs]
    preds_cat = cat(pred_list...; dims=3)  # (time, 2, ns)
    Ŷmean = mean(preds_cat, dims=3)[:,:,1]
    lower = mapslices(x -> quantile(x, 0.025), preds_cat; dims=3)[:,:,1]
    upper = mapslices(x -> quantile(x, 0.975), preds_cat; dims=3)[:,:,1]
    mse, mae, r2 = metrics(Ŷmean, Y)
    coverage = mean((Y .>= lower) .& (Y .<= upper))
    return mse, mae, r2, coverage
end

mse_train, mae_train, r2_train, cov_train = evaluate_split(t_train, Y_train, u0; nsamples=50)
@info "Train metrics" mse_train mae_train r2_train cov_train

mse_val, mae_val, r2_val, cov_val = evaluate_split(t_val, Y_val, u0_val; nsamples=50)
@info "Val metrics" mse_val mae_val r2_val cov_val

mse_test, mae_test, r2_test, cov_test = evaluate_split(t_test, Y_test, u0_test; nsamples=50)
@info "Test metrics" mse_test mae_test r2_test cov_test

# ------------------------- Save results ---------------------------
mkpath(joinpath(@__DIR__, "..", "checkpoints"))
BSON.@save joinpath(@__DIR__, "..", "checkpoints", "bayesian_node_results.bson") chain mse_train mae_train r2_train cov_train mse_val mae_val r2_val cov_val mse_test mae_test r2_test cov_test n_samples max_points

println("Saved bayesian_node_results.bson (posterior + metrics)")

# Append to paper table
using DataFrames, CSV
out = DataFrame(Architecture=["Bayesian-NODE"],
    MSE_train=[mse_train], MAE_train=[mae_train], R2_train=[r2_train], Coverage_train=[cov_train],
    MSE_val=[mse_val], MAE_val=[mae_val], R2_val=[r2_val], Coverage_val=[cov_val],
    MSE_test=[mse_test], MAE_test=[mae_test], R2_test=[r2_test], Coverage_test=[cov_test])

out_path = joinpath(@__DIR__, "..", "paper", "results", "final_model_performance.csv")
mkpath(dirname(out_path));
CSV.write(out_path, out; append=true)

println("Bayesian NODE training + evaluation complete!") 