module Microgrid
# Simple microgrid-like toy system. Keep it readable.

using DifferentialEquations

"Control schedule: charge 0–6h, idle 6–18h, discharge 18–24h (repeat daily)."
control_input(t) = t % 24 < 6  ?  1.0 :
                   t % 24 < 18 ?  0.0 : -0.8

"Pgen(t): pretend solar PV peaking at noon."
generation(t) = max(0, sin((t - 6) * π / 12))

"Load(t): base 0.6 + small oscillation."
load(t) = 0.6 + 0.2 * sin(t * π / 12)

"demand(t): we can reuse load, or define separately."
demand(t) = load(t)

"""
microgrid!:

x1 = battery state of charge (SOC) in [0,1]
x2 = 'grid pressure/frequency' proxy (unitless)
p  = (ηin, ηout, α, β, γ)

dx1 = (charge/discharge with efficiency) - demand
dx2 = -α*x2 + β*(Pgen - Pload) + γ*x1
"""
function microgrid!(dx, x, p, t)
    x1, x2 = x
    ηin, ηout, α, β, γ = p

    u    = control_input(t)
    Pin  = u > 0 ? ηin * u : (1 / ηout) * u
    Pnet = generation(t) - load(t)

    dx[1] = Pin - demand(t)
    dx[2] = -α * x2 + β * Pnet + γ * x1
end

end # module
