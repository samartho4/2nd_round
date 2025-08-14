using DataFrames, Statistics, Random, StatsBase

"""
    augment_training_data(train_data, target_size=15000)

Expand training data through data augmentation and noise injection.
Current 1,500 points is insufficient for neural ODE training.
"""
function augment_training_data(train_data::DataFrame, target_size::Int=15000)
    println("Original training size: $(nrow(train_data))")

    cols = Set(Symbol.(names(train_data)))
    if !all(x -> x in cols, [:x1, :x2])
        error("augment_training_data requires columns :x1 and :x2")
    end

    augmented_parts = DataFrame[]
    push!(augmented_parts, train_data)  # Original data

    # Add Gaussian noise variants
    noise_levels = [0.01, 0.02, 0.05]
    for noise_level in noise_levels
        noisy = copy(train_data)
        try
            noisy.x1 .= noisy.x1 .+ noise_level .* randn(nrow(noisy)) .* (std(noisy.x1) + eps())
            noisy.x2 .= noisy.x2 .+ noise_level .* randn(nrow(noisy)) .* (std(noisy.x2) + eps())
            push!(augmented_parts, noisy)
        catch e
            println("  ⚠️  Noise augmentation failed at level $(noise_level): $e")
        end
    end

    combined = vcat(augmented_parts...)

    final_data = combined
    if nrow(combined) > target_size
        idx = StatsBase.sample(1:nrow(combined), target_size; replace=false)
        final_data = combined[idx, :]
    end

    println("Final training size: $(nrow(final_data))")
    return final_data
end 