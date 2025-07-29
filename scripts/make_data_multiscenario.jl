# Multi-Scenario Data Generator for Microgrid Project
# --------------------------------------------------
# This script supersedes `make_data.jl` for large-scale, publishable datasets.
# It reads `data/scenario_info_improved.csv`, solves the microgrid ODE for each
# scenario, samples noisy / irregular observations, and writes train / val /
# test CSVs per scenario **and** global concatenated CSVs.  The resulting dataset
# contains >20 000 samples across 10 scenarios by default.
#
# Usage (from project root):
#   julia --project . scripts/make_data_multiscenario.jl [n_obs_per_scenario]
#
# If `n_obs_per_scenario` is omitted, the default is 2_000 so 10×2_000 = 20 000.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using DifferentialEquations, CSV, DataFrames, Random, Statistics, Dates

# Optional: Plots – only if installed
try
    @eval import Plots
catch
    @warn "Plots.jl not available – skipping figure generation"
end

include(joinpath(@__DIR__, "..", "src", "Microgrid.jl"))
using .Microgrid

# ---------------------------- Helper Functions --------------------------------

"""Parse a string like "(0.9, 0.9, 0.3, 1.2, 0.4)" into a NTuple{5,Float64}."""
function parse_tuple(str::AbstractString)
    Meta.parse(str) |> eval |> Tuple
end

"""Parse a string like "[0.5, 0.0]" into a Vector{Float64}."""
function parse_vector(str::AbstractString)
    Meta.parse(str) |> eval |> collect
end

function generate_single_scenario(id::AbstractString, p, u0::Vector{Float64};
        n_obs::Int, noise_x1::Float64, noise_x2::Float64, dropout::Float64)
    # 1. Solve dense truth on [0,72]h (3 days) with 0.05h resolution
    tspan        = (0.0, 72.0)
    saveat_dense = 0.05
    prob   = ODEProblem(Microgrid.microgrid!, u0, tspan, p)
    sol    = solve(prob, Tsit5(), saveat=saveat_dense)
    t_true = sol.t
    X_true = hcat(sol.u...)'

    # 2. Irregular observation times
    Random.seed!(42 + hash(id))   # scenario-specific seed for reproducibility
    t_obs  = sort(rand(n_obs) .* (tspan[2] - tspan[1]))
    t_obs[1] = 0.0
    X_obs  = [sol(t) for t in t_obs] |> x -> hcat(x...)'

    # 3. Add heteroscedastic noise
    X_noisy = copy(X_obs)
    X_noisy[:,1] .+= noise_x1 .* randn(n_obs)
    X_noisy[:,2] .+= noise_x2 .* randn(n_obs)

    # 4. Randomly drop some points
    mask    = rand(n_obs) .> dropout
    t_keep  = t_obs[mask]
    X_keep  = X_noisy[mask, :]

    # 5. Temporal split (same as original script)
    t_train_end = 48.0
    t_val_end   = 60.0

    train_idx = findall(t_keep .<= t_train_end)
    val_idx   = findall((t_keep .> t_train_end) .& (t_keep .<= t_val_end))
    test_idx  = findall(t_keep .> t_val_end)

    to_df(t, X) = DataFrame(time=t, x1=X[:,1], x2=X[:,2])

    df_train = to_df(t_keep[train_idx], X_keep[train_idx, :])
    df_val   = to_df(t_keep[val_idx],   X_keep[val_idx,   :])
    df_test  = to_df(t_keep[test_idx],  X_keep[test_idx,  :])
    df_true  = DataFrame(time=t_true, x1=X_true[:,1], x2=X_true[:,2])

    return df_train, df_val, df_test, df_true
end

# ---------------------------- Main Workflow -----------------------------------

function main()
    # CLI arg: observations per scenario
    n_obs_per_scenario = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 2_000

    info_path = joinpath(@__DIR__, "..", "data", "scenario_info_improved.csv")
    info = CSV.read(info_path, DataFrame)

    # Prepare output folders
    mkpath(joinpath(@__DIR__, "..", "data", "scenarios"))

    global_train = DataFrame(time=Float64[], x1=Float64[], x2=Float64[], scenario=String[])
    global_val   = similar(global_train)
    global_test  = similar(global_train)

    for row in eachrow(info)
        id        = row.Scenario
        p         = parse_tuple(row.Parameters)
        u0        = parse_vector(row.InitialConditions)
        noise_x1  = row.NoiseX1
        noise_x2  = row.NoiseX2
        dropout   = row.DropoutRate

        df_train, df_val, df_test, df_true = generate_single_scenario(id, p, u0;
            n_obs=n_obs_per_scenario,
            noise_x1=noise_x1, noise_x2=noise_x2, dropout=dropout)

        # Add scenario column for global concatenation
        df_train.scenario .= id
        df_val.scenario   .= id
        df_test.scenario  .= id

        append!(global_train, df_train)
        append!(global_val, df_val)
        append!(global_test, df_test)

        # Save per-scenario CSVs
        out_dir = joinpath(@__DIR__, "..", "data", "scenarios", id)
        mkpath(out_dir)
        CSV.write(joinpath(out_dir, "train.csv"), df_train)
        CSV.write(joinpath(out_dir, "val.csv"),   df_val)
        CSV.write(joinpath(out_dir, "test.csv"),  df_test)
        CSV.write(joinpath(out_dir, "true_dense.csv"), df_true)
    end

    # Save global concatenated splits
    CSV.write(joinpath(@__DIR__, "..", "data", "train_improved.csv"), global_train)
    CSV.write(joinpath(@__DIR__, "..", "data", "val_improved.csv"),   global_val)
    CSV.write(joinpath(@__DIR__, "..", "data", "test_improved.csv"),  global_test)

    println("\n✅ Generated $(nrow(global_train)+nrow(global_val)+nrow(global_test)) total observations across $(length(unique(global_train.scenario))) scenarios.")
    println("   → train=$(nrow(global_train)), val=$(nrow(global_val)), test=$(nrow(global_test))")
    println("   CSVs saved to data/scenarios/ and data/*_improved.csv")

    # Optional: quick sanity histogram of scenario counts
    if @isdefined Plots
        counts = combine(groupby(global_train, :scenario), nrow => :count)
        Plots.bar(counts.scenario, counts.count, xlabel="Scenario", ylabel="Train pts", title="Per-scenario train counts")
        Plots.savefig(joinpath(@__DIR__, "..", "figures", "improved_data_distribution.png"))
        println("   📊 Saved figures/improved_data_distribution.png")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__ && !isinteractive()
    main()
end 