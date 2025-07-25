# Step 3A: Train a Neural ODE with plain optimization (MAP-ish).
using Pkg; Pkg.activate(".")
using CSV, DataFrames, DifferentialEquations, Lux, DiffEqFlux, Optim, Zygote, Plots, Random, BSON, Optimisers

# --- Load data ---
df_train = CSV.read("data/train.csv", DataFrame)
df_val   = CSV.read("data/val.csv",   DataFrame)

t_train = df_train.time
Y_train = Matrix(df_train[:, [:x1, :x2]])

t_val = df_val.time
Y_val = Matrix(df_val[:, [:x1, :x2]])

u0_est = Y_train[1, :]
tspan  = (minimum(t_train), maximum(t_train))

# --- Define NN dynamics ---
nn = Lux.Chain(
    Lux.Dense(3, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)
ps, st = Lux.setup(Random.default_rng(), nn)

# Flatten parameters for optimization
flatθ, re = Optimisers.destructure(ps)

function nndyn!(dx, x, p, t)
    # p is flat; we need to reconstruct structured params
    ps_flat = re(p)
    inp = vcat(x, t)
    y, _ = nn(inp, st, ps_flat)
    dx .= y
end

prob = ODEProblem(nndyn!, u0_est, tspan, flatθ)

# Loss: MSE on train points
function loss(θ)
    sol = solve(prob, Tsit5(), p=θ, saveat=t_train, sensealg=InterpolatingAdjoint())
    Yhat = hcat(sol.u...)'
    return mean(abs2, Yhat .- Y_train)
end

println("Starting ADAM...")
res_adam = Optim.optimize(loss, flatθ, Optim.Adam(alpha=0.01), Optim.Options(iterations=500))
println("Switching to LBFGS...")
res_lbfgs = Optim.optimize(loss, Optim.minimizer(res_adam), Optim.LBFGS(), Optim.Options(iterations=300))

θ_map = Optim.minimizer(res_lbfgs)
final_loss = loss(θ_map)
println("✅ Final train MSE = ", final_loss)

# --- Eval on train & val ---
function predict(times, θ)
    sol = solve(prob, Tsit5(), p=θ, saveat=times, sensealg=InterpolatingAdjoint())
    hcat(sol.u...)'
end

Yhat_train = predict(t_train, θ_map)
Yhat_val   = predict(t_val,   θ_map)

mse_train = mean(abs2, Yhat_train .- Y_train)
mse_val   = mean(abs2, Yhat_val   .- Y_val)
println("Train MSE: $mse_train")
println("Val   MSE: $mse_val")

# --- Plots ---
mkpath("figures")
p1 = scatter(t_train, Y_train[:,1], ms=2, label="x1 train noisy", xlabel="time", ylabel="x1")
plot!(p1, t_train, Yhat_train[:,1], lw=2, label="x1 BNODE MAP")
p2 = scatter(t_train, Y_train[:,2], ms=2, label="x2 train noisy", xlabel="time", ylabel="x2")
plot!(p2, t_train, Yhat_train[:,2], lw=2, label="x2 BNODE MAP")
plot(p1, p2, layout=(2,1))
savefig("figures/bnode_map_train.png")
println("✅ Saved figures/bnode_map_train.png")

p3 = scatter(t_val, Y_val[:,1], ms=2, label="x1 val noisy", xlabel="time", ylabel="x1")
plot!(p3, t_val, Yhat_val[:,1], lw=2, label="x1 BNODE MAP")
p4 = scatter(t_val, Y_val[:,2], ms=2, label="x2 val noisy", xlabel="time", ylabel="x2")
plot!(p4, t_val, Yhat_val[:,2], lw=2, label="x2 BNODE MAP")
plot(p3, p4, layout=(2,1))
savefig("figures/bnode_map_val.png")
println("✅ Saved figures/bnode_map_val.png")

# --- Save params ---
mkpath("checkpoints")
BSON.@save "checkpoints/bnode_map.bson" θ_map
println("✅ Saved checkpoints/bnode_map.bson")

println("\nFeynman checks:\n",
        "1) Why include time t in the NN input?\n",
        "2) Name 2 reasons the solver might produce NaNs.\n",
        "3) Val error > train error. What besides 'bigger net' can you try?")
