# Step 3B: Bayesian NN-ODE with Turing (HMC/NUTS).
using Pkg; Pkg.activate(".")
using CSV, DataFrames, DifferentialEquations, Lux, DiffEqFlux, Zygote, Turing, Distributions, MCMCChains, Plots, Random, BSON, Statistics, Optimisers

# --- Load data (use train+val or just train at first) ---
df_train = CSV.read("data/train.csv", DataFrame)
t_obs = df_train.time
Y_obs = Matrix(df_train[:, [:x1, :x2]])

u0_est = Y_obs[1, :]
tspan  = (minimum(t_obs), maximum(t_obs))

# --- Rebuild same NN as MAP step ---
nn = Lux.Chain(
    Lux.Dense(3, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)
ps0, st = Lux.setup(Random.default_rng(), nn)
flatθ0, re = Optimisers.destructure(ps0)
D = length(flatθ0)

function nndyn!(dx, x, p, t)
    # p is structured ps; we closure st and nn
    inp = vcat(x, t)
    y, _ = nn(inp, st, p)
    dx .= y
end

@model function bnode_model(t, Y, u0)
    σ ~ truncated(Normal(0, 0.05), 0, Inf)        # obs noise scale
    θ ~ MvNormal(zeros(D), 1.0)                   # weight prior

    ps = re(θ)
    prob = ODEProblem(nndyn!, u0, (minimum(t), maximum(t)), ps)
    sol  = solve(prob, Tsit5(), saveat=t, sensealg=InterpolatingAdjoint();
                 abstol=1e-6, reltol=1e-6)

    Yhat = hcat(sol.u...)'
    for i in 1:size(Y,1)
        Y[i,:] ~ MvNormal(Yhat[i,:], σ * I(2))
    end
end

Turing.setadbackend(:zygote)
Turing.setrdcache(true)

model = bnode_model(t_obs, Y_obs, u0_est)

println("Sampling... (this can be slow)")
chain = sample(model, NUTS(0.65), 300; progress=true)
println(chain)

mkpath("checkpoints")
BSON.@save "checkpoints/bnode_chain.bson" chain

# --- Posterior predictive on test set ---
df_test = CSV.read("data/test.csv", DataFrame)
t_test  = df_test.time
Y_test  = Matrix(df_test[:, [:x1,:x2]])

n_samp = min(50, length(chain[:θ][:,1]))  # take first 50 samples
preds = Array{Float64}(undef, length(t_test), 2, n_samp)

for (k, row) in enumerate(eachrow(chain[:θ])[1:n_samp])
    ps_k = re(Vector(row))
    probk = ODEProblem(nndyn!, u0_est, (minimum(t_test), maximum(t_test)), ps_k)
    solk  = solve(probk, Tsit5(), saveat=t_test, sensealg=InterpolatingAdjoint())
    preds[:,:,k] = hcat(solk.u...)'
end

μ  = dropdims(mean(preds, dims=3), dims=3)
lo = dropdims(quantile(preds, 0.025, dims=3), dims=3)
hi = dropdims(quantile(preds, 0.975, dims=3), dims=3)

mkpath("figures")
p1 = scatter(t_test, Y_test[:,1], ms=3, label="x1 test noisy", xlabel="time", ylabel="x1")
plot!(p1, t_test, μ[:,1], lw=2, label="mean")
plot!(p1, t_test, lo[:,1], lw=1, ls=:dash, label="2.5%")
plot!(p1, t_test, hi[:,1], lw=1, ls=:dash, label="97.5%")
savefig("figures/bnode_bayes_x1.png")

p2 = scatter(t_test, Y_test[:,2], ms=3, label="x2 test noisy", xlabel="time", ylabel="x2")
plot!(p2, t_test, μ[:,2], lw=2, label="mean")
plot!(p2, t_test, lo[:,2], lw=1, ls=:dash, label="2.5%")
plot!(p2, t_test, hi[:,2], lw=1, ls=:dash, label="97.5%")
savefig("figures/bnode_bayes_x2.png")

println("✅ Saved figures/bnode_bayes_x1.png & _x2.png")
println("\nCheck R̂ and ESS in chain summary for convergence.")
