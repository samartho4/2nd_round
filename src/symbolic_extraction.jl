module SymbolicExtraction

export symbolify

using Statistics

"""
    symbolify(nn_outputs::AbstractVector, inputs::AbstractMatrix; library::Symbol=:poly, degree::Int=2)

Fits a simple polynomial regression of specified degree to map `inputs` → `nn_outputs`.
Returns a Dict with coefficients, intercept, feature_names, R2, and standardization.
Inputs are shaped as (N, D). For library=:poly, expands up to total degree.
"""
function symbolify(nn_outputs::AbstractVector, inputs::AbstractMatrix; library::Symbol=:poly, degree::Int=2)
    @assert library == :poly "Only :poly library is currently supported"
    N, D = size(inputs)
    @assert length(nn_outputs) == N

    # build polynomial feature names and design matrix up to given degree (no cross terms beyond pairwise if degree=2)
    feats = Vector{Vector{Float64}}(undef, N)
    feature_names = String[]

    function build_row(x::Vector{Float64})
        out = Float64[1.0]
        if isempty(feature_names)
            push!(feature_names, "1")
        end
        # degree 1
        for j in 1:D
            push!(out, x[j])
            if length(feature_names) < length(out)
                push!(feature_names, "x$j")
            end
        end
        if degree >= 2
            # squares
            for j in 1:D
                push!(out, x[j]^2)
                if length(feature_names) < length(out)
                    push!(feature_names, "x$j^2")
                end
            end
            # pairwise products
            for j in 1:D-1
                for k in j+1:D
                    push!(out, x[j]*x[k])
                    if length(feature_names) < length(out)
                        push!(feature_names, "x$j*x$k")
                    end
                end
            end
        end
        return out
    end

    for i in 1:N
        feats[i] = build_row(vec(inputs[i, :]))
    end

    Φ = reduce(vcat, (permutedims(v) for v in feats))
    y = collect(nn_outputs)

    μΦ = mean(Φ, dims=1)
    σΦ = std(Φ, dims=1) .+ 1e-8
    Φs = (Φ .- μΦ) ./ σΦ
    μy = mean(y)
    σy = std(y) + 1e-8
    ys = (y .- μy) ./ σy

    # ridge regression
    λ = 1e-3
    Ireg = Matrix{Float64}(I, size(Φs,2), size(Φs,2))
    βs = (Φs' * Φs .+ λ .* Ireg) \ (Φs' * ys)

    # map back
    β = (βs ./ vec(σΦ)) .* σy
    β0 = μy - sum((vec(μΦ) ./ vec(σΦ)) .* vec(βs)) * σy

    ŷ = Φ * β .+ β0
    ss_res = sum((y .- ŷ).^2)
    ss_tot = sum((y .- mean(y)).^2)
    R2 = 1 - ss_res / ss_tot

    return Dict(
        :coeffs => β,
        :intercept => β0,
        :feature_names => feature_names,
        :R2 => R2,
        :n_points => N,
        :standardization => Dict(:mu => vec(μΦ), :sigma => vec(σΦ), :mu_y => μy, :sigma_y => σy),
    )
end

end # module 