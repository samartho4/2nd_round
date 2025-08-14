# Final Results Summary Script
# Evaluates models by simulating trajectories and comparing to ground truth
# Generates clean Markdown table for paper submission

using CSV, DataFrames, BSON, Statistics, DifferentialEquations, Printf, TOML
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures
using .Microgrid

# Config
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
const OUT_RES_DIR = joinpath(@__DIR__, "..", String(getcfg("outputs/results", :paths, :results_dir)))
mkpath(OUT_RES_DIR)
mkpath(joinpath(@__DIR__, "..", "paper", "results"))

# Standard solver tolerances
a_tol = getcfg(1e-8, :solver, :abstol)
r_tol = getcfg(1e-8, :solver, :reltol)

println("FINAL RESULTS SUMMARY GENERATION")
println("="^60)

# Load test dataset
println("Loading test dataset...")
df_test = CSV.read("data/test_dataset.csv", DataFrame)
println("✅ Test data loaded: $(nrow(df_test)) points")

# Get unique scenarios for trajectory simulation
scenarios = unique(df_test.scenario)
println("📊 Found $(length(scenarios)) test scenarios: $(scenarios)")

# Load all three models
println("\nLoading trained models...")

# 1. Bayesian Neural ODE
bayesian_file = BSON.load("checkpoints/bayesian_neural_ode_results.bson")
bayesian_results = bayesian_file[:bayesian_results]
bayesian_params = bayesian_results[:params_mean]
arch_name = haskey(bayesian_results, :arch) ? String(bayesian_results[:arch]) : "baseline"
function pick_arch(arch::AbstractString)
    a = lowercase(String(arch))
    if a == "baseline"
        return (:baseline, baseline_nn!, 10)
    elseif a == "baseline_bias"
        return (:baseline_bias, baseline_nn_bias!, 14)
    elseif a == "deep"
        return (:deep, deep_nn!, 26)
    else
        return (:baseline, baseline_nn!, 10)
    end
end
arch_sym, bayes_deriv_fn, _ = pick_arch(arch_name)
println("✅ Bayesian Neural ODE loaded: arch=$(arch_sym), $(length(bayesian_params)) parameters")

# 2. UDE (Universal Differential Equations)
ude_file = BSON.load("checkpoints/ude_results_fixed.bson")
ude_results = ude_file[:ude_results]
physics_params = ude_results[:physics_params_mean]
neural_params = ude_results[:neural_params_mean]
println("✅ UDE loaded: $(length(physics_params)) physics + $(length(neural_params)) neural parameters")

# 3. Symbolic UDE Extraction
symbolic_file = BSON.load("checkpoints/symbolic_models_fixed.bson")
symbolic_results = symbolic_file[:symbolic_results]
symbolic_r2 = symbolic_results[:avg_r2]
println("✅ Symbolic extraction loaded: R2 = $(round(symbolic_r2, digits=4))")

# Function to simulate trajectory using Bayesian Neural ODE
function simulate_bayesian_trajectory(x0, tspan, params)
    function bayesian_dynamics!(dx, x, p, t)
        bayes_deriv_fn(dx, x, p, t)
    end
    prob = ODEProblem(bayesian_dynamics!, x0, tspan, params)
    sol = solve(prob, Tsit5(), abstol=a_tol, reltol=r_tol, maxiters=10000)
    return sol
end

# Function to simulate trajectory using UDE
function simulate_ude_trajectory(x0, tspan, physics_p, neural_p)
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
        h1 = tanh(nn_params[1]*x1 + nn_params[2]*x2 + nn_params[3]*Pgen + nn_params[4]*Pload + nn_params[5]*t + nn_params[6])
        h2 = tanh(nn_params[7]*x1 + nn_params[8]*x2 + nn_params[9]*Pgen + nn_params[10]*Pload + nn_params[11]*t + nn_params[12])
        nn_output = nn_params[13]*h1 + nn_params[14]*h2 + nn_params[15]
        dx[2] = -α * x2 + nn_output + γ * x1
    end
    p_ude = [physics_p..., neural_p...]
    prob = ODEProblem(ude_dynamics!, x0, tspan, p_ude)
    sol = solve(prob, Tsit5(), abstol=a_tol, reltol=r_tol, maxiters=10000)
    return sol
end

# Function to simulate trajectory using physics-only model
function simulate_physics_trajectory(x0, tspan)
    p_physics = (0.9, 0.9, 0.3, 1.2, 0.4)
    prob = ODEProblem(Microgrid.microgrid!, x0, tspan, p_physics)
    sol = solve(prob, Tsit5(), abstol=a_tol, reltol=r_tol, maxiters=10000)
    return sol
end

# Function to evaluate a single scenario
function evaluate_scenario(scenario, df_test, bayesian_params, physics_params, neural_params)
    println("\n📊 Evaluating scenario: $(scenario)")
    scenario_data = filter(row -> row.scenario == scenario, df_test)
    t_data = Array(scenario_data.time)
    Y_data = Matrix(scenario_data[:, [:x1, :x2]])
    x0 = Y_data[1, :]
    tspan = (t_data[1], t_data[end])
    println("   Time span: $(round(tspan[1], digits=2)) to $(round(tspan[2], digits=2))")
    println("   Data points: $(length(t_data))")

    results = Dict(
        "bayesian_mse" => NaN,
        "ude_mse" => NaN,
        "physics_mse" => NaN,
        "points" => 0
    )

    try
        bayesian_sol = simulate_bayesian_trajectory(x0, tspan, bayesian_params)
        bayesian_traj = bayesian_sol(t_data)
        bayesian_mse = mean((bayesian_traj .- Y_data').^2)
        results["bayesian_mse"] = bayesian_mse
        println("   Bayesian Neural ODE MSE: $(round(bayesian_mse, digits=2))")

        ude_sol = simulate_ude_trajectory(x0, tspan, physics_params, neural_params)
        ude_traj = ude_sol(t_data)
        ude_mse = mean((ude_traj .- Y_data').^2)
        results["ude_mse"] = ude_mse
        println("   UDE MSE: $(round(ude_mse, digits=2))")

        physics_sol = simulate_physics_trajectory(x0, tspan)
        physics_traj = physics_sol(t_data)
        physics_mse = mean((physics_traj .- Y_data').^2)
        results["physics_mse"] = physics_mse
        println("   Physics-only MSE: $(round(physics_mse, digits=2))")

        results["points"] = length(t_data)
    catch e
        println("   ⚠️  Simulation failed for $(scenario): $(e)")
    end

    return results
end

# Evaluate models on trajectory simulation
println("\n" * "="^60)
println("TRAJECTORY SIMULATION EVALUATION")
println("="^60)

eval_scenarios = scenarios[1:min(3, length(scenarios))]  # Use first 3 scenarios
println("Evaluating on scenarios: $(eval_scenarios)")

global bayesian_mse_total = 0.0
global ude_mse_total = 0.0
global physics_mse_total = 0.0
global total_points = 0

for scenario in eval_scenarios
    results = evaluate_scenario(scenario, df_test, bayesian_params, physics_params, neural_params)
    if !isnan(results["bayesian_mse"])
        global bayesian_mse_total += results["bayesian_mse"] * results["points"]
        global ude_mse_total += results["ude_mse"] * results["points"]
        global physics_mse_total += results["physics_mse"] * results["points"]
        global total_points += results["points"]
    end
end

# Calculate final metrics
if total_points > 0
    bayesian_final_mse = bayesian_mse_total / total_points
    ude_final_mse = ude_mse_total / total_points
    physics_final_mse = physics_mse_total / total_points

    println("\n" * "="^60)
    println("FINAL TRAJECTORY SIMULATION RESULTS")
    println("="^60)
    println("Total evaluation points: $(total_points)")
    println("Scenarios evaluated: $(length(eval_scenarios))")
    println()
    println("📊 TRAJECTORY MSE RESULTS:")
    println("   Bayesian Neural ODE: $(round(bayesian_final_mse, digits=2))")
    println("   UDE (Universal Differential Equations): $(round(ude_final_mse, digits=2))")
    println("   Physics-Only Model: $(round(physics_final_mse, digits=2))")
    println()
    println("🎯 SYMBOLIC DISCOVERY RESULTS:")
    println("   UDE Neural Network R2: $(round(symbolic_r2, digits=4))")
else
    println("❌ No successful simulations completed")
    bayesian_final_mse = NaN
    ude_final_mse = NaN
    physics_final_mse = NaN
end

# Generate final Markdown table
println("\n" * "="^60)
println("FINAL RESULTS TABLE (MARKDOWN)")
println("="^60)

markdown_table = """
## Final Results

| Method | Trajectory MSE | Symbolic R2 | Training Data | Numerical Stability |
|--------|----------------|-------------|---------------|-------------------|
| Bayesian Neural ODE | $(round(bayesian_final_mse, digits=2)) | N/A | 1,500 points | abstol=$(a_tol), reltol=$(r_tol) |
| UDE (Universal Differential Equations) | $(round(ude_final_mse, digits=2)) | $(round(symbolic_r2, digits=4)) | 1,500 points | abstol=$(a_tol), reltol=$(r_tol) |
| Physics-Only Model | $(round(physics_final_mse, digits=2)) | N/A | N/A | abstol=$(a_tol), reltol=$(r_tol) |
| Symbolic Discovery | N/A | $(round(symbolic_r2, digits=4)) | N/A | N/A |

**Key Findings:**
- **Trajectory Simulation**: Models evaluated by simulating full trajectories and comparing to ground truth
- **Physics Discovery**: Symbolic surrogate fits UDE neural residual (R2 = $(round(symbolic_r2, digits=4))); physics validation checked separately
- **Numerical Stability**: All simulations use strict tolerances (abstol=$(a_tol), reltol=$(r_tol))
- **Evaluation**: $(total_points) points across $(length(eval_scenarios)) scenarios
"""

println(markdown_table)

# Save results to file
println("\n💾 Saving results...")
open(joinpath(OUT_RES_DIR, "final_results_table.md"), "w") do io
    write(io, markdown_table)
end
cp(joinpath(OUT_RES_DIR, "final_results_table.md"), joinpath(@__DIR__, "..", "paper", "results", "final_results_table.md"); force=true)
println("✅ Results saved to $(OUT_RES_DIR)/final_results_table.md and mirrored to paper/results/")

# Load OOD dataset if present
has_ood = isfile("data/ood_test_dataset.csv")
if has_ood
    println("\nDetected OOD dataset: evaluating OOD trajectory MSE as well")
    df_ood = CSV.read("data/ood_test_dataset.csv", DataFrame)
    ood_scenarios = unique(df_ood.scenario)
    println("📊 OOD scenarios: $(ood_scenarios)")

    function evaluate_block(df_block)
        t_data = Array(df_block.time)
        Y_data = Matrix(df_block[:, [:x1, :x2]])
        x0 = Y_data[1, :]
        tspan = (t_data[1], t_data[end])
        res = Dict("points"=>0, "bayesian_mse"=>NaN, "ude_mse"=>NaN, "physics_mse"=>NaN)
        try
            bsol = simulate_bayesian_trajectory(x0, tspan, bayesian_params)
            usol = simulate_ude_trajectory(x0, tspan, physics_params, neural_params)
            psol = simulate_physics_trajectory(x0, tspan)
            btraj = bsol(t_data); utraj = usol(t_data); ptraj = psol(t_data)
            res["bayesian_mse"] = mean((btraj .- Y_data').^2)
            res["ude_mse"]      = mean((utraj .- Y_data').^2)
            res["physics_mse"]  = mean((ptraj .- Y_data').^2)
            res["points"]       = length(t_data)
        catch e
            println("   ⚠️  OOD simulation failed: $(e)")
        end
        return res
    end

    ood_total = Dict("bayesian"=>0.0, "ude"=>0.0, "physics"=>0.0, "points"=>0)
    for scn in ood_scenarios
        blk = filter(row -> row.scenario == scn, df_ood)
        r = evaluate_block(blk)
        if !isnan(r["bayesian_mse"]) 
            ood_total["bayesian"] += r["bayesian_mse"] * r["points"]
            ood_total["ude"]      += r["ude_mse"]      * r["points"]
            ood_total["physics"]  += r["physics_mse"]  * r["points"]
            ood_total["points"]   += r["points"]
        end
    end

    if ood_total["points"] > 0
        ood_b = ood_total["bayesian"] / ood_total["points"]
        ood_u = ood_total["ude"]      / ood_total["points"]
        ood_p = ood_total["physics"]  / ood_total["points"]
        println("\nOOD TRAJECTORY SIMULATION RESULTS")
        println("   Bayesian Neural ODE: $(round(ood_b, digits=2))")
        println("   UDE: $(round(ood_u, digits=2))")
        println("   Physics-only: $(round(ood_p, digits=2))")
        # Append to markdown table
        markdown_table *= "\n## OOD Results\n\n"
        markdown_table *= "| Method | OOD Trajectory MSE |\n|--------|-------------------|\n"
        markdown_table *= "| Bayesian Neural ODE | $(round(ood_b, digits=2)) |\n"
        markdown_table *= "| UDE (Universal Differential Equations) | $(round(ood_u, digits=2)) |\n"
        markdown_table *= "| Physics-Only Model | $(round(ood_p, digits=2)) |\n"
    end
end

println("\n✅ FINAL RESULTS SUMMARY COMPLETE")
println("="^60) 