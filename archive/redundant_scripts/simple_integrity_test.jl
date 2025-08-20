#!/usr/bin/env julia

println("🔬 SIMPLE RESEARCH INTEGRITY TESTING")
println("=" ^ 50)

# Test 1: Data File Existence and Basic Properties
println("\n📊 TEST 1: DATA FILE INTEGRITY")
println("-" ^ 30)

data_files = [
    "data/training_dataset.csv",
    "data/validation_dataset.csv", 
    "data/test_dataset.csv"
]

for file in data_files
    if isfile(file)
        size_bytes = filesize(file)
        println("✅ $file exists ($(size_bytes) bytes)")
        
        # Quick check of file content
        open(file, "r") do io
            first_line = readline(io)
            println("   Header: $first_line")
            
            # Count lines
            line_count = countlines(file)
            println("   Lines: $line_count")
        end
    else
        println("❌ $file missing")
    end
end

# Test 2: Model File Existence
println("\n🤖 TEST 2: MODEL FILE INTEGRITY")
println("-" ^ 30)

model_files = [
    "checkpoints/bayesian_neural_ode_results.bson",
    "checkpoints/ude_results_fixed.bson"
]

for file in model_files
    if isfile(file)
        size_bytes = filesize(file)
        println("✅ $file exists ($(size_bytes) bytes)")
    else
        println("❌ $file missing")
    end
end

# Test 3: Results File Existence
println("\n📈 TEST 3: RESULTS FILE INTEGRITY")
println("-" ^ 30)

results_files = [
    "results/current_performance_summary.md",
    "results/neurips_comprehensive_analysis.md",
    "results/quantitative_model_comparison.md",
    "results/simple_model_comparison.csv"
]

for file in results_files
    if isfile(file)
        size_bytes = filesize(file)
        println("✅ $file exists ($(size_bytes) bytes)")
    else
        println("❌ $file missing")
    end
end

# Test 4: Script File Existence
println("\n🔧 TEST 4: SCRIPT FILE INTEGRITY")
println("-" ^ 30)

script_files = [
    "scripts/train.jl",
    "scripts/evaluate.jl",
    "scripts/simple_model_comparison.jl",
    "scripts/expand_existing_data.jl"
]

for file in script_files
    if isfile(file)
        size_bytes = filesize(file)
        println("✅ $file exists ($(size_bytes) bytes)")
    else
        println("❌ $file missing")
    end
end

# Test 5: Data Leakage Check (Simple)
println("\n🔍 TEST 5: DATA LEAKAGE CHECK")
println("-" ^ 30)

# Read scenario information from files
function extract_scenarios(filename)
    scenarios = Set{String}()
    if isfile(filename)
        open(filename, "r") do io
            # Skip header
            readline(io)
            for line in eachline(io)
                if contains(line, ",")
                    parts = split(line, ",")
                    if length(parts) >= 4
                        scenario = strip(parts[4])
                        push!(scenarios, scenario)
                    end
                end
            end
        end
    end
    return scenarios
end

train_scenarios = extract_scenarios("data/training_dataset.csv")
val_scenarios = extract_scenarios("data/validation_dataset.csv")
test_scenarios = extract_scenarios("data/test_dataset.csv")

println("Training scenarios: $(length(train_scenarios))")
println("Validation scenarios: $(length(val_scenarios))")
println("Test scenarios: $(length(test_scenarios))")

# Check for overlap
train_test_overlap = intersect(train_scenarios, test_scenarios)
if length(train_test_overlap) > 0
    println("❌ CRITICAL: Data leakage detected!")
    println("   Overlapping scenarios: $train_test_overlap")
else
    println("✅ No data leakage detected")
end

# Test 6: Documentation Consistency
println("\n📚 TEST 6: DOCUMENTATION CONSISTENCY")
println("-" ^ 30)

# Check if claimed numbers match actual
if isfile("data/training_dataset.csv")
    train_lines = countlines("data/training_dataset.csv")
    actual_train_samples = train_lines - 1  # Subtract header
    
    claimed_train_samples = 7334
    
    println("Claimed training samples: $claimed_train_samples")
    println("Actual training samples: $actual_train_samples")
    
    if claimed_train_samples == actual_train_samples
        println("✅ Training sample count consistent")
    else
        println("❌ INCONSISTENCY: Training sample count mismatch!")
    end
end

# Test 7: Performance Claims Check
println("\n📊 TEST 7: PERFORMANCE CLAIMS CHECK")
println("-" ^ 30)

if isfile("results/simple_model_comparison.csv")
    println("✅ Performance results file exists")
    
    # Read and check performance values
    open("results/simple_model_comparison.csv", "r") do io
        lines = readlines(io)
        if length(lines) > 1
            println("   Performance data available")
            
            # Check for extreme values
            for line in lines[2:end]  # Skip header
                if contains(line, "BNN-ODE") && contains(line, "0.000006")
                    println("   ⚠️  WARNING: Extremely low BNN-ODE MSE detected")
                end
            end
        end
    end
else
    println("❌ Performance results file missing")
end

# Test 8: Critical Issues Summary
println("\n🚨 CRITICAL ISSUES SUMMARY")
println("-" ^ 30)

issues_found = 0

# Check for data leakage
if length(train_test_overlap) > 0
    println("❌ CRITICAL: Data leakage detected")
    issues_found += 1
end

# Check for missing critical files
critical_files = [
    "data/training_dataset.csv",
    "data/validation_dataset.csv", 
    "data/test_dataset.csv",
    "checkpoints/bayesian_neural_ode_results.bson",
    "checkpoints/ude_results_fixed.bson"
]

for file in critical_files
    if !isfile(file)
        println("❌ CRITICAL: Missing file: $file")
        issues_found += 1
    end
end

if issues_found == 0
    println("✅ No critical issues found")
else
    println("❌ $issues_found critical issues found")
end

println("\n" ^ 50)
println("🔬 SIMPLE RESEARCH INTEGRITY TEST COMPLETE")
println("=" ^ 50) 