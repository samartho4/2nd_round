module NeuralNODEArchitectures
# Basic Neural ODE architectures used throughout the project.
# Each derivative function follows the DifferentialEquations.jl signature:
#    f!(dx, x, p, t)
# so it can be passed directly to `ODEProblem`.
#
# The neural parameters `p` are assumed to be a Vector{Float64}.  For the
# baseline & PINN we use the original 10-parameter 3→2→2 network.
# The attention version uses 16 parameters for a single-head dot-product.
# -----------------------------------------------------------------------------

import ..Microgrid

export baseline_nn!, pinn_nn!, attention_nn!, ensemble_predict

# 1. Baseline Neural ODE (same as in TECHNICAL_ANALYSIS.md)
function baseline_nn!(dx, x, p::AbstractVector, t)
    # Input vector [x1, x2, t]
    inp1, inp2 = x
    inp3 = t

    # Hidden layer (tanh)
    h1 = tanh(p[1]*inp1 + p[2]*inp2 + p[3]*inp3)
    h2 = tanh(p[4]*inp1 + p[5]*inp2 + p[6]*inp3)

    # Output layer (linear)
    dx[1] = p[7]*h1 + p[8]*h2
    dx[2] = p[9]*h1 + p[10]*h2
end

# 2. Neural ODE
# Adds a residual term of the physical microgrid model (weighted by λ)
function pinn_nn!(dx, x, p_aug::AbstractVector, t)
    # p_aug = [θ (10); λ (1)]
    θ = view(p_aug, 1:10)
    λ = p_aug[11]

    # Neural component
    baseline_nn!(dx, x, θ, t)
    dx_nn = copy(dx)

    # Physical residual (reuse Microgrid.microgrid!)
    micro_dx = similar(dx)
    # Hard-coded physical params (could be optimisable later)
    p_phys = (0.9, 0.9, 0.3, 1.2, 0.4)
    Microgrid.microgrid!(micro_dx, x, p_phys, t)

    # Combine
    @. dx = dx_nn + λ * micro_dx
end

# 3. Attention-based Neural ODE
function attention_nn!(dx, x, p::AbstractVector, t)
    # Parameter layout:
    #   W_q (3), W_k (3), W_v (3)  – first 9
    #   Output weights: 2×1 vector w (2) + bias (2)  – next 4 ⇒ total 13
    #   Extra nonlinear proj layer weights (3) ⇒ 16 total (kept small)
    # Simple sigmoid helper
    σ(z) = 1 / (1 + exp(-z))

    q = [p[1]*x[1] + p[2]*x[2] + p[3]*t]
    k = [p[4]*x[1] + p[5]*x[2] + p[6]*t]
    v = [p[7]*x[1] + p[8]*x[2] + p[9]*t]

    attn = v * σ(q*k)  # simple scalar attention

    # Project to derivatives
    dx[1] = p[10]*attn + p[12]
    dx[2] = p[11]*attn + p[13]
end

# 4. Simple ensemble prediction helper
auto_mean(vs...) = reduce(+, vs) ./ length(vs)
function ensemble_predict(pred_funcs::Vector, x0, tspan)
    preds = [f() for f in pred_funcs]  # each f returns Matrix (length(t),2)
    return auto_mean(preds...)
end

end # module 