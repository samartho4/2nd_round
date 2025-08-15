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

export baseline_nn!, baseline_nn_bias!, deep_nn!, pinn_nn!, attention_nn!, ensemble_predict, ude_nn_forward
export mc_forward, summarize_ensemble

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

# 1b. Baseline with biases (3→2→2, 14 params)
# Parameter layout:
#  - Hidden neuron 1: w11,w12,w13,b1  (p[1:4])
#  - Hidden neuron 2: w21,w22,w23,b2  (p[5:8])
#  - Output dx1: v11,v12,b3           (p[9:11])
#  - Output dx2: v21,v22,b4           (p[12:14])
function baseline_nn_bias!(dx, x, p::AbstractVector, t)
    inp1, inp2 = x
    inp3 = t

    h1 = tanh(p[1]*inp1 + p[2]*inp2 + p[3]*inp3 + p[4])
    h2 = tanh(p[5]*inp1 + p[6]*inp2 + p[7]*inp3 + p[8])

    dx[1] = p[9]*h1 + p[10]*h2 + p[11]
    dx[2] = p[12]*h1 + p[13]*h2 + p[14]
end

# 1c. Deeper network with biases (3→4→2, 26 params)
# Layout:
#  - Hidden (4 units): 4×3 weights (12) + 4 biases = 16  ⇒ p[1:16]
#  - Output (2 units): 2×4 weights (8) + 2 biases = 10   ⇒ p[17:26]
function deep_nn!(dx, x, p::AbstractVector, t)
    x1, x2 = x
    inp = (x1, x2, t)

    # Hidden 4
    w = p[1:12]
    b = p[13:16]
    h = NTuple{4,Float64}(
        (
            tanh(w[1]*inp[1] + w[2]*inp[2] + w[3]*inp[3] + b[1]),
            tanh(w[4]*inp[1] + w[5]*inp[2] + w[6]*inp[3] + b[2]),
            tanh(w[7]*inp[1] + w[8]*inp[2] + w[9]*inp[3] + b[3]),
            tanh(w[10]*inp[1] + w[11]*inp[2] + w[12]*inp[3] + b[4])
        )
    )

    wo = p[17:24]
    bo1, bo2 = p[25], p[26]
    dx[1] = wo[1]*h[1] + wo[2]*h[2] + wo[3]*h[3] + wo[4]*h[4] + bo1
    dx[2] = wo[5]*h[1] + wo[6]*h[2] + wo[7]*h[3] + wo[8]*h[4] + bo2
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

# Centralized UDE NN forward (15 params as used in scripts)
# Input = (x1, x2, Pgen, Pload, t)
# Hidden: two tanh neurons with biases; Output: linear with bias
function ude_nn_forward(x1::T, x2::T, Pgen, Pload, t, nn_params::AbstractVector) where T
    h1 = tanh(nn_params[1]*x1 + nn_params[2]*x2 + nn_params[3]*Pgen + nn_params[4]*Pload + nn_params[5]*t + nn_params[6])
    h2 = tanh(nn_params[7]*x1 + nn_params[8]*x2 + nn_params[9]*Pgen + nn_params[10]*Pload + nn_params[11]*t + nn_params[12])
    return nn_params[13]*h1 + nn_params[14]*h2 + nn_params[15]
end

# 4. Simple ensemble prediction helper
auto_mean(vs...) = reduce(+, vs) ./ length(vs)
function ensemble_predict(pred_funcs::Vector, x0, tspan)
    preds = [f() for f in pred_funcs]  # each f returns Matrix (length(t),2)
    return auto_mean(preds...)
end

# 5. Monte Carlo forward utility for epistemic uncertainty via stochastic masking
#    This is a lightweight surrogate for MC dropout. Caller passes a masking function.
function mc_forward(deriv_fn::Function, x::AbstractVector, p::AbstractVector, t::Real; mask_fn::Function, n::Int=50)
	preds = Vector{Vector{Float64}}(undef, n)
	for i in 1:n
		pm = mask_fn(p)
		dx = zeros(length(x))
		deriv_fn(dx, x, pm, t)
		preds[i] = collect(dx)
	end
	return preds
end

function summarize_ensemble(samples::Vector{<:AbstractVector})
	# returns (mean, std) per dimension
	n = length(samples)
	if n == 0
		return (Float64[], Float64[])
	end
	m = length(samples[1])
	acc = zeros(m)
	for s in samples
		acc .+= s
	end
	μ = acc ./ n
	σ = zeros(m)
	for s in samples
		σ .+= (s .- μ) .^ 2
	end
	σ = sqrt.(σ ./ max(1, n - 1))
	return (μ, σ)
end

end # module 