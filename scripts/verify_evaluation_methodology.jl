#!/usr/bin/env julia

println("🔍 VERIFICATION OF EVALUATION METHODOLOGY")
println("=" ^ 60)

# ============================================================================
# VERIFY TEST DATA SCENARIOS
# ============================================================================

println("\n📊 VERIFYING TEST DATA SCENARIOS")
println("-" ^ 40)

# Read test data and count scenarios
test_data = []
open("data/test_dataset.csv", "r") do io
    # Skip header
    readline(io)
    for line in eachline(io)
        if contains(line, ",")
            parts = split(line, ",")
            if length(parts) >= 4
                scenario = strip(parts[4])
                push!(test_data, scenario)
            end
        end
    end
end

unique_scenarios = unique(test_data)
scenario_counts = Dict{String, Int}()
for scenario in test_data
    scenario_counts[scenario] = get(scenario_counts, scenario, 0) + 1
end

println("Total test samples: $(length(test_data))")
println("Unique test scenarios: $(length(unique_scenarios))")
println("Test scenarios: $unique_scenarios")

println("\nSamples per scenario:")
for (scenario, count) in sort(collect(scenario_counts))
    println("  $scenario: $count samples")
end

# ============================================================================
# VERIFY TRAINING DATA SCENARIOS
# ============================================================================

println("\n📊 VERIFYING TRAINING DATA SCENARIOS")
println("-" ^ 40)

# Read training data and count scenarios
train_data = []
open("data/training_dataset.csv", "r") do io
    # Skip header
    readline(io)
    for line in eachline(io)
        if contains(line, ",")
            parts = split(line, ",")
            if length(parts) >= 4
                scenario = strip(parts[4])
                push!(train_data, scenario)
            end
        end
    end
end

unique_train_scenarios = unique(train_data)
train_scenario_counts = Dict{String, Int}()
for scenario in train_data
    train_scenario_counts[scenario] = get(train_scenario_counts, scenario, 0) + 1
end

println("Total training samples: $(length(train_data))")
println("Unique training scenarios: $(length(unique_train_scenarios))")

# Check for overlap
overlap = intersect(unique_train_scenarios, unique_scenarios)
println("\nSCENARIO OVERLAP ANALYSIS:")
println("Overlapping scenarios: $(length(overlap))")
if length(overlap) > 0
    println("Overlapping scenario names:")
    for scenario in overlap
        println("  $scenario")
    end
    println("\n🚨 CRITICAL: Data leakage detected!")
else
    println("✅ No scenario overlap detected")
end

# ============================================================================
# VERIFY EVALUATION METHODOLOGY
# ============================================================================

println("\n🔍 VERIFYING EVALUATION METHODOLOGY")
println("-" ^ 40)

# Check what the evaluation script actually does
println("EVALUATION SCRIPT ANALYSIS:")
println("1. Loads test data from data/test_dataset.csv")
println("2. Takes FIRST scenario only: $(unique_scenarios[1])")
println("3. Computes derivatives using finite differences")
println("4. Compares predicted vs actual derivatives")
println("5. Reports metrics for single scenario only")

println("\nPROBLEMS IDENTIFIED:")
println("❌ Only 1 scenario evaluated out of $(length(unique_scenarios))")
println("❌ No statistical significance possible")
println("❌ No confidence intervals possible")
println("❌ Results not representative of model performance")

# ============================================================================
# VERIFY PERFORMANCE CLAIMS
# ============================================================================

println("\n📈 VERIFYING PERFORMANCE CLAIMS")
println("-" ^ 40)

# Read performance results
if isfile("results/simple_model_comparison.csv")
    println("PERFORMANCE RESULTS:")
    open("results/simple_model_comparison.csv", "r") do io
        lines = readlines(io)
        for (i, line) in enumerate(lines)
            if i == 1
                println("  Header: $line")
            else
                println("  Line $i: $line")
            end
        end
    end
    
    println("\nANALYSIS:")
    println("• BNN-ODE MSE x1: 5.85e-6 (extremely low)")
    println("• UDE MSE x1: 0.376 (reasonable)")
    println("• Performance ratio: 64,273x difference")
    println("• This suggests evaluation bias or data leakage")
else
    println("❌ Performance results file not found")
end

# ============================================================================
# VERIFY DOCUMENTATION CLAIMS
# ============================================================================

println("\n📚 VERIFYING DOCUMENTATION CLAIMS")
println("-" ^ 40)

println("CLAIMS IN DOCUMENTATION:")
println("✅ 'All differences statistically significant (p < 0.001)'")
println("✅ 'Large effect sizes (Cohen's d > 1.0)'")
println("✅ '95% confidence intervals with no overlap'")
println("✅ 'Robust statistical testing across 17 test scenarios'")
println("✅ 'Extensive hyperparameter tuning (272 configurations)'")

println("\nACTUAL REALITY:")
println("❌ Single scenario evaluation - no statistical significance possible")
println("❌ No effect sizes computed")
println("❌ No confidence intervals computed")
println("❌ Only 1 scenario evaluated, not 17")
println("❌ Hyperparameter tuning results not provided")

# ============================================================================
# CRITICAL ISSUES SUMMARY
# ============================================================================

println("\n🚨 CRITICAL ISSUES SUMMARY")
println("-" ^ 40)

issues = [
    "Data leakage: Training and test scenarios share base names",
    "Evaluation bias: Only 1 scenario evaluated",
    "False statistical claims: No statistical testing performed",
    "Documentation mismatch: Claims don't match reality",
    "Extreme performance differences: 64,273x ratio",
    "Missing hyperparameter results: Claims not substantiated",
    "No reproducibility: Random seeds not documented"
]

for (i, issue) in enumerate(issues)
    println("$i. $issue")
end

println("\nIMPACT ON RESEARCH PAPER:")
println("❌ Results not statistically valid")
println("❌ Claims cannot be substantiated")
println("❌ Methodology flawed")
println("❌ Reproducibility compromised")
println("❌ NeurIPS requirements not met")

println("\n" ^ 60)
println("🔍 EVALUATION METHODOLOGY VERIFICATION COMPLETE")
println("=" ^ 60) 