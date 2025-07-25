# Step 2: Generate noisy, irregular observations and split train/val/test.
using Pkg; Pkg.activate(".")
using DifferentialEquations, CSV, DataFrames, Random, Statistics, Plots
include(joinpath(@__DIR__, "..", "src", "Microgrid.jl"))
using .Microgrid

mkpath("data"); mkpath("figures")

# --- 1. Solve clean truth on dense grid ---
p    = (0.9, 0.9, 0.3, 1.2, 0.4)
u0   = [0.5, 0.0]
tspan = (0.0, 72.0)
saveat_dense = 0.05

prob = ODEProblem(Microgrid.microgrid!, u0, tspan, p)
sol  = solve(prob, Tsit5(), saveat=saveat_dense)

t_true = sol.t
X_true = hcat(sol.u...)'

# --- 2. Irregular sample times ---
Random.seed!(42)
n_obs = 400
t_obs = sort(rand(n_obs) .* (tspan[2] - tspan[1]))  # uniform 0..72
t_obs[1] = 0.0  # ensure we have t=0

# Interpolate solution at t_obs
X_obs = [sol(t) for t in t_obs] |> x -> hcat(x...)'  # (n_obs, 2)

# --- 3. Add noise ---
σx1, σx2 = 0.02, 0.03
X_noisy = copy(X_obs)
X_noisy[:,1] .+= σx1 .* randn(n_obs)
X_noisy[:,2] .+= σx2 .* randn(n_obs)

# --- 4. Drop 10% randomly (missing data) ---
mask = rand(n_obs) .> 0.10
t_keep = t_obs[mask]
X_keep = X_noisy[mask, :]

# --- 5. Temporal split ---
t_train_end = 48.0
t_val_end   = 60.0

train_idx = findall(t_keep .<= t_train_end)
val_idx   = findall((t_keep .> t_train_end) .& (t_keep .<= t_val_end))
test_idx  = findall(t_keep .> t_val_end)

to_df(t, X) = DataFrame(time=t, x1=X[:,1], x2=X[:,2])

df_train = to_df(t_keep[train_idx], X_keep[train_idx, :])
df_val   = to_df(t_keep[val_idx],   X_keep[val_idx,   :])
df_test  = to_df(t_keep[test_idx],  X_keep[test_idx,  :])

CSV.write("data/train.csv", df_train)
CSV.write("data/val.csv",   df_val)
CSV.write("data/test.csv",  df_test)
CSV.write("data/true_dense.csv", DataFrame(time=t_true, x1=X_true[:,1], x2=X_true[:,2]))

println("✅ Saved CSVs to /data")
println("Counts: train=$(nrow(df_train)), val=$(nrow(df_val)), test=$(nrow(df_test))")

# --- 6. Plot sanity ---
p1 = plot(t_true, X_true[:,1], lw=2, label="x1 true", xlabel="time (h)", ylabel="x1")
scatter!(p1, df_train.time, df_train.x1, ms=2, label="x1 train noisy")
p2 = plot(t_true, X_true[:,2], lw=2, label="x2 true", xlabel="time (h)", ylabel="x2")
scatter!(p2, df_train.time, df_train.x2, ms=2, label="x2 train noisy")

plot(p1, p2, layout=(2,1))
savefig("figures/noisy_vs_true.png")
println("✅ Saved figures/noisy_vs_true.png")

println("\nSanity Qs:\n",
        "1) Why split by time, not random shuffle?\n",
        "2) If σx1=0.2, what happens to training?\n",
        "3) Why keep true_dense.csv?\n",
        "4) If a whole 6h window is missing, what issue arises?")
