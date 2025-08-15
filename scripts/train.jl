using TOML
include(joinpath(@__DIR__, "..", "src", "training.jl"))
using .Training

function parse_args(argv)
    # supports: --modeltype=bnn|ude  --seed=INT
    opts = Dict{String,Any}("modeltype"=>"both")
    for a in argv
        if startswith(a, "--modeltype=")
            opts["modeltype"] = split(a, "=", limit=2)[2]
        elseif startswith(a, "--seed=")
            opts["seed"] = parse(Int, split(a, "=", limit=2)[2])
        end
    end
    return opts
end

const CONFIG_PATH = joinpath(@__DIR__, "..", "config", "config.toml")
cfg = if isfile(CONFIG_PATH)
    try
        TOML.parsefile(CONFIG_PATH)
    catch
        Dict{String,Any}()
    end
else
    Dict{String,Any}()
end

opts = parse_args(ARGS)
if haskey(opts, "seed")
    cfg["train"] = get(cfg, "train", Dict{String,Any}())
    cfg["train"]["seed"] = opts["seed"]
end

mkpath(joinpath(@__DIR__, "..", "checkpoints"))

mt = String(get(opts, "modeltype", "both"))
if mt == "bnn" || mt == "both"
    println("Training Bayesian Neural ODE ...")
    Training.train!(; modeltype=:bnn, cfg=cfg)
end
if mt == "ude" || mt == "both"
    println("Training UDE ...")
    Training.train!(; modeltype=:ude, cfg=cfg)
end

println("Done.") 