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

using DifferentialEquations, CSV, DataFrames, Random, Statistics, Dates, TOML, SHA

# Optional: Plots – only if installed
try
    @eval import Plots
catch
    @warn "Plots.jl not available – skipping figure generation"
end

include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
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

    cfg_path = joinpath(@__DIR__, "..", "config", "config.toml")
    cfg = isfile(cfg_path) ? TOML.parsefile(cfg_path) : Dict{String,Any}()
    getcfg(dflt, ks...) = begin
        v = cfg
        for k in ks
            if v isa Dict && haskey(v, String(k))
                v = v[String(k)]
            else
                return dflt
            end
        end
        return v
    end

    info_path = joinpath(@__DIR__, "..", "data", "scenario_info_improved.csv")
    if isfile(info_path)
        info = CSV.read(info_path, DataFrame)
    else
        info = nothing
        @warn "scenario_info_improved.csv not found; falling back to assembling from data/scenarios/*"
    end

    # Prepare output folders
    mkpath(joinpath(@__DIR__, "..", "data", "scenarios"))

    global_train = DataFrame(time=Float64[], x1=Float64[], x2=Float64[], scenario=String[])
    global_val   = similar(global_train)
    global_test  = similar(global_train)

    if info !== nothing
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
    else
        # Fallback: assemble global CSVs from existing per-scenario files
        scen_dir = joinpath(@__DIR__, "..", "data", "scenarios")
        if isdir(scen_dir)
            for scn in readdir(scen_dir)
                sdir = joinpath(scen_dir, scn)
                if isdir(sdir) && isfile(joinpath(sdir, "train.csv")) && isfile(joinpath(sdir, "val.csv")) && isfile(joinpath(sdir, "test.csv"))
                    df_tr = CSV.read(joinpath(sdir, "train.csv"), DataFrame); df_tr.scenario .= scn; append!(global_train, df_tr)
                    df_va = CSV.read(joinpath(sdir, "val.csv"), DataFrame);   df_va.scenario .= scn; append!(global_val, df_va)
                    df_te = CSV.read(joinpath(sdir, "test.csv"), DataFrame);  df_te.scenario .= scn; append!(global_test, df_te)
                end
            end
        else
            error("No scenarios directory found at $(scen_dir)")
        end
    end

    # Optionally enforce scenario-disjoint global splits
    scenario_disjoint = getcfg(false, :data, :scenario_disjoint)
    if scenario_disjoint
        train_scn = Set(getcfg(String[], :data, :train_scenarios))
        test_scn  = Set(getcfg(String[], :data, :test_scenarios))
        if isempty(train_scn) || isempty(test_scn)
            # Deterministic scenario split: first 80% train, last 20% test
            scns = unique(global_train.scenario)
            sort!(scns)
            ntr = Int(floor(0.8 * length(scns)))
            train_scn = Set(scns[1:ntr])
            test_scn  = Set(scns[ntr+1:end])
        end
        global_train = filter(r -> r.scenario in train_scn, global_train)
        global_val   = filter(r -> !(r.scenario in test_scn), global_val) # keep val non-test by default
        global_test  = filter(r -> r.scenario in test_scn, global_test)
    end

    # Save global concatenated splits
    CSV.write(joinpath(@__DIR__, "..", "data", "training_dataset.csv"), global_train)
    CSV.write(joinpath(@__DIR__, "..", "data", "validation_dataset.csv"),   global_val)
    CSV.write(joinpath(@__DIR__, "..", "data", "test_dataset.csv"),  global_test)

    println("\n✅ Generated $(nrow(global_train)+nrow(global_val)+nrow(global_test)) total observations across $(length(unique(global_train.scenario))) scenarios.")
    println("   → train=$(nrow(global_train)), val=$(nrow(global_val)), test=$(nrow(global_test))")
    println("   CSVs saved to data/scenarios/ and data/*_dataset.csv")

    # Optional: quick sanity histogram of scenario counts
    if @isdefined Plots
        counts = combine(groupby(global_train, :scenario), nrow => :count)
        Plots.bar(counts.scenario, counts.count, xlabel="Scenario", ylabel="Train pts", title="Per-scenario train counts")
        mkpath(joinpath(@__DIR__, "..", "figures"))
        Plots.savefig(joinpath(@__DIR__, "..", "figures", "improved_data_distribution.png"))
        println("   📊 Saved figures/improved_data_distribution.png")
    end

    # Write checksums for integrity
    try
        outp = IOBuffer()
        function write_hash(io, path)
            bytes = read(path)
            h = bytes2hex(sha256(bytes))
            println(io, "$(h)  $(path)")
        end
        # Hash top-level CSVs
        for f in filter(f->endswith(f, ".csv"), readdir(joinpath(@__DIR__, "..", "data")))
            write_hash(outp, joinpath("data", f))
        end
        # Hash per-scenario CSVs
        scen_dir = joinpath(@__DIR__, "..", "data", "scenarios")
        for scn in readdir(scen_dir)
            for f in filter(f->endswith(f, ".csv"), readdir(joinpath(scen_dir, scn)))
                write_hash(outp, joinpath("data", "scenarios", scn, f))
            end
        end
        open(joinpath(@__DIR__, "..", "data", "hashes.txt"), "w") do io
            write(io, String(take!(outp)))
        end
        println("   🔒 Wrote data/hashes.txt")
    catch e
        @warn "Failed to write hashes: $e"
    end
end

main() 