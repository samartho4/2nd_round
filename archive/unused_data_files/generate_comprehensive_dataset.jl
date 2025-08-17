#!/usr/bin/env julia

"""
    generate_comprehensive_dataset.jl

Generate comprehensive microgrid datasets with much larger sample sizes
and better utilization of all available scenarios.

This script addresses the data underutilization issue by:
1. Generating more scenarios with diverse parameters
2. Creating longer time series with more data points
3. Ensuring proper train/test overlap with larger datasets
4. Providing multiple dataset sizes for different training needs
"""

using Random, Dates, CSV, DataFrames, Printf
using DifferentialEquations, Statistics, LinearAlgebra, Plots
using TOML, SHA

# Include the realistic physics module
include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
using .Microgrid

"""
    create_comprehensive_scenarios()

Create a much larger set of diverse scenarios for comprehensive training.
"""
function create_comprehensive_scenarios()
    println("🔬 Creating comprehensive scenario set...")
    
    scenarios = Dict{String,Dict{String,Any}}()
    
    # Simplified parameter combinations for reliable generation
    ηin_values = [0.85, 0.88, 0.90, 0.92, 0.95]
    ηout_values = [0.85, 0.88, 0.90, 0.92, 0.95]
    α_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    β_values = [0.8, 1.0, 1.2, 1.4]
    γ_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    # Initial condition ranges
    soc_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    power_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    scenario_counter = 1
    
    # Generate scenarios with controlled combinations
    for ηin in ηin_values
        for ηout in ηout_values
            for α in α_values
                for β in β_values
                    for γ in γ_values
                        # Take every 10th combination to avoid too many scenarios
                        if scenario_counter % 10 == 0
                            for soc in soc_values
                                for power in power_values
                                    if scenario_counter <= 50  # Limit to 50 scenarios
                                        scenario_id = "C$(scenario_counter)"
                                        
                                        scenarios[scenario_id] = Dict{String,Any}(
                                            "params" => [ηin, ηout, α, β, γ],
                                            "initial" => [soc, power],
                                            "name" => "Comprehensive-$(scenario_counter)",
                                            "description" => "ηin=$(ηin), ηout=$(ηout), α=$(α), β=$(β), γ=$(γ), SOC=$(soc), P=$(power)"
                                        )
                                        
                                        scenario_counter += 1
                                    end
                                end
                            end
                        end
                        scenario_counter += 1
                    end
                end
            end
        end
    end
    
    println("  ✅ Created $(length(scenarios)) comprehensive scenarios")
    return scenarios
end

"""
    generate_extended_trajectory(scenario_id, params, u0, tspan, noise_level=0.02)

Generate extended trajectories with more data points and longer time spans.
"""
function generate_extended_trajectory(scenario_id::String, params::Vector{Float64}, 
                                    u0::Vector{Float64}, tspan::Tuple{Float64,Float64}, 
                                    noise_level::Float64=0.02)
    
    # Use finer time resolution for more data points
    prob = ODEProblem(microgrid_ode!, u0, tspan, params)
    sol = solve(prob, Rodas4(), reltol=1e-4, abstol=1e-6, saveat=0.1, maxiters=10000)
    
    # Validate physics constraints
    violations = validate_physics(sol)
    if !isempty(violations)
        @warn "Physics violations in $scenario_id" violations
    end
    
    # Add realistic measurement noise
    Random.seed!(hash(scenario_id))
    
    n_points = length(sol.t)
    noise_x1 = noise_level * 0.05 * randn(n_points)  # 5% SOC noise
    noise_x2 = noise_level * 2.0 * randn(n_points)   # 2 kW power noise
    
    # Create trajectory dataframe
    trajectory = DataFrame(
        time = sol.t,
        x1 = sol[1, :] .+ noise_x1,  # SOC with noise
        x2 = sol[2, :] .+ noise_x2,  # Power imbalance with noise
        scenario = scenario_id,
        P_gen = [generation(t, params[5]) for t in sol.t],
        P_load = [load(t, params[4]) for t in sol.t]
    )
    
    # Apply physical constraints after noise
    trajectory.x1 = clamp.(trajectory.x1, 0.0, 1.0)
    
    return trajectory, violations
end

"""
    create_comprehensive_splits(scenarios)

Create comprehensive train/validation/test splits with extended time spans.
"""
function create_comprehensive_splits(scenarios::Dict)
    println("🕒 Creating comprehensive temporal splits...")
    
    all_trajectories = DataFrame[]
    validation_summary = Dict{String,Any}()
    
    # Extended time spans for more data
    train_span = (0.0, 72.0)      # 3 days
    val_span = (60.0, 120.0)      # Overlap: 60-72h, extend to 120h  
    test_span = (108.0, 168.0)    # Overlap: 108-120h, extend to 168h
    
    validation_summary["temporal_splits"] = Dict(
        "train" => train_span,
        "validation" => val_span,
        "test" => test_span,
        "overlap_train_val" => (60.0, 72.0),
        "overlap_val_test" => (108.0, 120.0)
    )
    
    for (scenario_id, scenario_data) in scenarios
        params = scenario_data["params"]
        u0 = scenario_data["initial"]
        
        if scenario_counter(scenario_id) % 10 == 0
            println("  → Generating $scenario_id: $(scenario_data["description"])")
        end
        
        # Generate trajectories for each split
        train_traj, train_viol = generate_extended_trajectory("$(scenario_id)-train", params, u0, train_span, 0.02)
        val_traj, val_viol = generate_extended_trajectory("$(scenario_id)-val", params, u0, val_span, 0.02)  
        test_traj, test_viol = generate_extended_trajectory("$(scenario_id)-test", params, u0, test_span, 0.02)
        
        # Mark splits
        train_traj[!, :split] .= "train"
        val_traj[!, :split] .= "validation" 
        test_traj[!, :split] .= "test"
        
        push!(all_trajectories, train_traj, val_traj, test_traj)
        
        # Track violations
        validation_summary["$(scenario_id)_violations"] = Dict(
            "train" => train_viol,
            "validation" => val_viol, 
            "test" => test_viol
        )
    end
    
    combined_data = vcat(all_trajectories...)
    
    # Verify overlapping distributions
    verify_comprehensive_overlap(combined_data, validation_summary)
    
    return combined_data, validation_summary
end

"""
    scenario_counter(scenario_id)

Extract scenario number from scenario ID.
"""
function scenario_counter(scenario_id::String)
    return parse(Int, replace(scenario_id, "C" => ""))
end

"""
    verify_comprehensive_overlap(data, summary)

Ensure comprehensive train/validation/test have overlapping distributions.
"""
function verify_comprehensive_overlap(data::DataFrame, summary::Dict)
    println("🔍 Verifying comprehensive train/test distribution overlap...")
    
    train_data = filter(row -> row.split == "train", data)
    val_data = filter(row -> row.split == "validation", data)
    test_data = filter(row -> row.split == "test", data)
    
    # Check x1 (SOC) overlap
    x1_train = (minimum(train_data.x1), maximum(train_data.x1))
    x1_val = (minimum(val_data.x1), maximum(val_data.x1))
    x1_test = (minimum(test_data.x1), maximum(test_data.x1))
    
    # Check x2 (Power) overlap  
    x2_train = (minimum(train_data.x2), maximum(train_data.x2))
    x2_val = (minimum(val_data.x2), maximum(val_data.x2))
    x2_test = (minimum(test_data.x2), maximum(test_data.x2))
    
    # Calculate overlap percentages
    x1_overlap_val = calculate_overlap(x1_train, x1_val)
    x1_overlap_test = calculate_overlap(x1_train, x1_test)
    x2_overlap_val = calculate_overlap(x2_train, x2_val)
    x2_overlap_test = calculate_overlap(x2_train, x2_test)
    
    summary["distribution_analysis"] = Dict(
        "x1_ranges" => Dict("train" => x1_train, "val" => x1_val, "test" => x1_test),
        "x2_ranges" => Dict("train" => x2_train, "val" => x2_val, "test" => x2_test),
        "x1_overlap" => Dict("train_val" => x1_overlap_val, "train_test" => x1_overlap_test),
        "x2_overlap" => Dict("train_val" => x2_overlap_val, "train_test" => x2_overlap_test)
    )
    
    println("  ✅ x1 (SOC) ranges: Train$(x1_train), Val$(x1_val), Test$(x1_test)")
    println("  ✅ x2 (Power) ranges: Train$(x2_train), Val$(x2_val), Test$(x2_test)")
    println("  ✅ x1 overlap: Train-Val $(round(x1_overlap_val*100,digits=1))%, Train-Test $(round(x1_overlap_test*100,digits=1))%")
    println("  ✅ x2 overlap: Train-Val $(round(x2_overlap_val*100,digits=1))%, Train-Test $(round(x2_overlap_test*100,digits=1))%")
    
    # Ensure minimum overlap
    min_overlap = 0.3  # Require at least 30% overlap
    if x1_overlap_test < min_overlap || x2_overlap_test < min_overlap
        error("❌ Insufficient train/test overlap! x1: $(x1_overlap_test), x2: $(x2_overlap_test)")
    end
    
    println("  ✅ Comprehensive distribution overlap validation PASSED")
end

"""
    calculate_overlap(range1, range2)

Calculate the fractional overlap between two ranges.
"""
function calculate_overlap(range1::Tuple{Float64,Float64}, range2::Tuple{Float64,Float64})
    min1, max1 = range1
    min2, max2 = range2
    
    overlap_min = max(min1, min2)
    overlap_max = min(max1, max2)
    
    if overlap_min >= overlap_max
        return 0.0  # No overlap
    end
    
    overlap_size = overlap_max - overlap_min
    range1_size = max1 - min1
    range2_size = max2 - min2
    
    # Return overlap as fraction of smaller range
    smaller_range_size = min(range1_size, range2_size)
    return overlap_size / smaller_range_size
end

"""
    save_comprehensive_datasets(data, validation_summary)

Save comprehensive datasets with multiple size options.
"""
function save_comprehensive_datasets(data::DataFrame, validation_summary::Dict)
    println("💾 Saving comprehensive datasets...")
    
    # Split data
    train_data = filter(row -> row.split == "train", data)
    val_data = filter(row -> row.split == "validation", data)
    test_data = filter(row -> row.split == "test", data)
    
    # Core columns
    core_columns = [:time, :x1, :x2, :scenario]
    
    # Save full comprehensive datasets
    CSV.write("data/training_dataset_comprehensive.csv", select(train_data, core_columns))
    CSV.write("data/validation_dataset_comprehensive.csv", select(val_data, core_columns))
    CSV.write("data/test_dataset_comprehensive.csv", select(test_data, core_columns))
    
    # Create different size subsets for different training needs
    sizes = [1000, 2000, 5000, 10000]
    
    for size in sizes
        if nrow(train_data) >= size
            # Take first 'size' samples from each split
            train_subset = train_data[1:size, :]
            val_subset = val_data[1:min(size, nrow(val_data)), :]
            test_subset = test_data[1:min(size, nrow(test_data)), :]
            
            CSV.write("data/training_dataset_$(size).csv", select(train_subset, core_columns))
            CSV.write("data/validation_dataset_$(size).csv", select(val_subset, core_columns))
            CSV.write("data/test_dataset_$(size).csv", select(test_subset, core_columns))
        end
    end
    
    # Update main datasets to use comprehensive data
    CSV.write("data/training_dataset.csv", select(train_data, core_columns))
    CSV.write("data/validation_dataset.csv", select(val_data, core_columns))
    CSV.write("data/test_dataset.csv", select(test_data, core_columns))
    
    # Save comprehensive validation summary
    open("data/comprehensive_generation_metadata.toml", "w") do f
        TOML.print(f, Dict(
            "generation_timestamp" => string(now()),
            "dataset_type" => "comprehensive",
            "physics_model" => "Realistic microgrid with battery SOC and power balance",
            "state_variables" => Dict(
                "x1" => "Battery State of Charge [0-1]",
                "x2" => "Power Imbalance [kW]"
            ),
            "parameters" => Dict(
                "ηin" => "Battery charging efficiency [0.85-0.95]",
                "ηout" => "Battery discharging efficiency [0.85-0.95]",
                "α" => "Grid coupling coefficient [0.1-0.5]",
                "β" => "Load response coefficient [0.8-1.5]",
                "γ" => "Generation variability [0.2-0.6]"
            ),
            "validation" => validation_summary,
            "data_integrity" => "VERIFIED - Comprehensive overlapping distributions",
            "available_sizes" => sizes
        ))
    end
    
    # Generate comprehensive scenario descriptions
    scenarios = create_comprehensive_scenarios()
    scenario_df = DataFrame(
        Scenario = collect(keys(scenarios)),
        Name = [s["name"] for s in values(scenarios)],
        Parameters = [string(s["params"]) for s in values(scenarios)],
        InitialConditions = [string(s["initial"]) for s in values(scenarios)],
        Description = [s["description"] for s in values(scenarios)]
    )
    CSV.write("data/comprehensive_scenario_descriptions.csv", scenario_df)
    
    # Generate data hash for integrity verification
    data_content = string(train_data) * string(val_data) * string(test_data)
    data_hash = bytes2hex(sha256(data_content))
    
    open("data/comprehensive_hashes.txt", "w") do f
        println(f, "# Comprehensive Dataset integrity hashes")
        println(f, "training_dataset_comprehensive.csv: $(bytes2hex(sha256(string(train_data))))")
        println(f, "validation_dataset_comprehensive.csv: $(bytes2hex(sha256(string(val_data))))")
        println(f, "test_dataset_comprehensive.csv: $(bytes2hex(sha256(string(test_data))))")
        println(f, "combined_hash: $data_hash")
        println(f, "generation_date: $(now())")
    end
    
    # Print comprehensive summary statistics
    println("📊 Comprehensive Dataset Generation Summary:")
    println("  → Training samples: $(nrow(train_data))")
    println("  → Validation samples: $(nrow(val_data))") 
    println("  → Test samples: $(nrow(test_data))")
    println("  → Total scenarios: $(length(unique(data.scenario)))")
    println("  → Time span: $(minimum(data.time)) - $(maximum(data.time)) hours")
    println("  → Available subset sizes: $sizes")
    println("  → Data integrity hash: $data_hash")
    println("  ✅ All comprehensive datasets saved")
end

"""
    main()

Main comprehensive data generation pipeline.
"""
function main()
    println("🔬 COMPREHENSIVE DATA GENERATION PIPELINE")
    println("=" ^ 50)
    
    # Set global seed for reproducibility
    Random.seed!(42)
    
    # Create comprehensive scenarios
    scenarios = create_comprehensive_scenarios()
    println("📋 Created $(length(scenarios)) comprehensive scenarios")
    
    # Generate comprehensive temporally-aware datasets
    data, validation_summary = create_comprehensive_splits(scenarios)
    
    # Save everything with documentation
    save_comprehensive_datasets(data, validation_summary)
    
    println("\n✅ COMPREHENSIVE DATA GENERATION COMPLETE")
    println("   → Much larger dataset sizes generated")
    println("   → Multiple subset sizes available for different training needs")
    println("   → Valid train/test overlap verified")  
    println("   → Comprehensive validation documented")
    println("   → Data integrity hashes computed")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 