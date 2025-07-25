# Step 1: Solve clean ODE and plot.
using Pkg; Pkg.activate(".")
using DifferentialEquations, Plots
include(joinpath(@__DIR__, "..", "src", "Microgrid.jl"))
using .Microgrid

# --- Parameters & initial state ---
p    = (0.9, 0.9, 0.3, 1.2, 0.4)  # (ηin, ηout, α, β, γ) -- just guesses
u0   = [0.5, 0.0]                 # start half-charged, neutral grid
tspan = (0.0, 72.0)               # 3 days

prob = ODEProblem(Microgrid.microgrid!, u0, tspan, p)
sol  = solve(prob, Tsit5(), saveat=0.1)

mkpath("figures")
plot(sol.t, hcat(sol.u...)', label=["x1 (SOC)" "x2 (grid proxy)"], xlabel="time (h)")
savefig("figures/ode_states.png")
println("✅ Saved figures/ode_states.png")

# Feynman checks in terminal:
println("Quick sanity prompts:\n",
        "1) If ηin doubles, what happens to x1 during charge?\n",
        "2) If α is huge, what happens to x2's oscillations?\n",
        "3) What does β multiply physically?")
