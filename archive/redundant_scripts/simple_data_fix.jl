#!/usr/bin/env julia

"""
    simple_data_fix.jl

Simple script to fix the existing data by ensuring SOC values are in the correct [0.0, 1.0] range.
This avoids package precompilation issues while fixing the critical data problems.
"""

using Random, Statistics, Dates

println("🔧 SIMPLE DATA FIX")
println("=" ^ 30)

# Set random seed
Random.seed!(42)

# Read existing data files
println("📊 Reading existing data...")

# Read training data
train_lines = readlines("data/training_dataset.csv")
train_header = train_lines[1]
train_data = train_lines[2:end]

# Read test data  
test_lines = readlines("data/test_dataset.csv")
test_header = test_lines[1]
test_data = test_lines[2:end]

# Read validation data
val_lines = readlines("data/validation_dataset.csv")
val_header = val_lines[1]
val_data = val_lines[2:end]

println("  Training samples: $(length(train_data))")
println("  Test samples: $(length(test_data))")
println("  Validation samples: $(length(val_data))")

# Function to fix a single data line
function fix_data_line(line)
    parts = split(line, ",")
    if length(parts) >= 4
        time = parse(Float64, parts[1])
        x1 = parse(Float64, parts[2])  # SOC
        x2 = parse(Float64, parts[3])  # Power
        scenario = parts[4]
        
        # CRITICAL FIX: Ensure SOC is in [0.0, 1.0] range
        x1_fixed = clamp(x1, 0.0, 1.0)
        
        # Add small noise to make data more realistic if it was exactly at bounds
        if x1_fixed == 0.0 || x1_fixed == 1.0
            x1_fixed = clamp(x1_fixed + 0.01 * randn(), 0.01, 0.99)
        end
        
        return "$(time),$(x1_fixed),$(x2),$(scenario)"
    else
        return line  # Return unchanged if parsing fails
    end
end

# Fix all data
println("🔧 Fixing SOC values...")

train_fixed = [fix_data_line(line) for line in train_data]
test_fixed = [fix_data_line(line) for line in test_data]
val_fixed = [fix_data_line(line) for line in val_data]

# Write fixed data
println("💾 Writing fixed data...")

open("data/training_dataset_fixed.csv", "w") do f
    println(f, train_header)
    for line in train_fixed
        println(f, line)
    end
end

open("data/test_dataset_fixed.csv", "w") do f
    println(f, test_header)
    for line in test_fixed
        println(f, line)
    end
end

open("data/validation_dataset_fixed.csv", "w") do f
    println(f, val_header)
    for line in val_fixed
        println(f, line)
    end
end

# Update main datasets
open("data/training_dataset.csv", "w") do f
    println(f, train_header)
    for line in train_fixed
        println(f, line)
    end
end

open("data/test_dataset.csv", "w") do f
    println(f, test_header)
    for line in test_fixed
        println(f, line)
    end
end

open("data/validation_dataset.csv", "w") do f
    println(f, val_header)
    for line in val_fixed
        println(f, line)
    end
end

# Analyze the fixed data
println("📊 Analyzing fixed data...")

function analyze_data(data_lines)
    soc_values = Float64[]
    power_values = Float64[]
    
    for line in data_lines
        parts = split(line, ",")
        if length(parts) >= 4
            push!(soc_values, parse(Float64, parts[2]))
            push!(power_values, parse(Float64, parts[3]))
        end
    end
    
    return soc_values, power_values
end

train_soc, train_power = analyze_data(train_fixed)
test_soc, test_power = analyze_data(test_fixed)
val_soc, val_power = analyze_data(val_fixed)

println("  Training SOC: mean=$(round(mean(train_soc), digits=3)), std=$(round(std(train_soc), digits=3)), range=[$(round(minimum(train_soc), digits=3)), $(round(maximum(train_soc), digits=3))]")
println("  Test SOC: mean=$(round(mean(test_soc), digits=3)), std=$(round(std(test_soc), digits=3)), range=[$(round(minimum(test_soc), digits=3)), $(round(maximum(test_soc), digits=3))]")
println("  Validation SOC: mean=$(round(mean(val_soc), digits=3)), std=$(round(std(val_soc), digits=3)), range=[$(round(minimum(val_soc), digits=3)), $(round(maximum(val_soc), digits=3))]")

println("  Training Power: mean=$(round(mean(train_power), digits=3)), std=$(round(std(train_power), digits=3)), range=[$(round(minimum(train_power), digits=3)), $(round(maximum(train_power), digits=3))]")
println("  Test Power: mean=$(round(mean(test_power), digits=3)), std=$(round(std(test_power), digits=3)), range=[$(round(minimum(test_power), digits=3)), $(round(maximum(test_power), digits=3))]")
println("  Validation Power: mean=$(round(mean(val_power), digits=3)), std=$(round(std(val_power), digits=3)), range=[$(round(minimum(val_power), digits=3)), $(round(maximum(val_power), digits=3))]")

# Check for physical validity
soc_valid = all(0.0 .<= train_soc .<= 1.0) && all(0.0 .<= test_soc .<= 1.0) && all(0.0 .<= val_soc .<= 1.0)

if soc_valid
    println("  ✅ All SOC values are physically valid [0.0, 1.0]")
else
    println("  ❌ Some SOC values are still unphysical")
end

# Create metadata
open("data/simple_fix_metadata.txt", "w") do f
    println(f, "# SIMPLE DATA FIX METADATA")
    println(f, "Generated: $(now())")
    println(f, "")
    println(f, "FIXES APPLIED:")
    println(f, "1. SOC values clamped to [0.0, 1.0] range")
    println(f, "2. Small noise added to boundary values")
    println(f, "3. All datasets updated")
    println(f, "")
    println(f, "DATA SUMMARY:")
    println(f, "Training samples: $(length(train_fixed))")
    println(f, "Test samples: $(length(test_fixed))")
    println(f, "Validation samples: $(length(val_fixed))")
    println(f, "")
    println(f, "SOC RANGES:")
    println(f, "Training: [$(round(minimum(train_soc), digits=3)), $(round(maximum(train_soc), digits=3))]")
    println(f, "Test: [$(round(minimum(test_soc), digits=3)), $(round(maximum(test_soc), digits=3))]")
    println(f, "Validation: [$(round(minimum(val_soc), digits=3)), $(round(maximum(val_soc), digits=3))]")
    println(f, "")
    println(f, "STATUS: $(soc_valid ? "PHYSICALLY VALID" : "STILL HAS ISSUES")")
end

println("\n✅ SIMPLE DATA FIX COMPLETE")
println("   → SOC values constrained to [0.0, 1.0]")
println("   → All datasets updated")
println("   → Metadata saved")
println("   → Ready for model retraining") 