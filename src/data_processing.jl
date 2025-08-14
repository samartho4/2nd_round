using Random, DataFrames, CSV

"""
    create_temporal_splits(scenario_files; train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

Creates proper temporal splits within each scenario to prevent data leakage.
CRITICAL: No future information can leak into training data.
Input:
- `scenario_files`: Vector of file paths to per-scenario complete time series (e.g., true_dense.csv)
Output:
- `(final_train::DataFrame, final_val::DataFrame, final_test::DataFrame)` with in-split next-step labels `x1_next`, `x2_next`.
"""
function create_temporal_splits(scenario_files; train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    Random.seed!(seed)

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8
        error("Ratios must sum to 1.0")
    end

    all_train = DataFrame[]
    all_val = DataFrame[]
    all_test = DataFrame[]

    for scenario_file in scenario_files
        println("Processing scenario: $scenario_file")
        data = DataFrame()
        try
            data = CSV.read(scenario_file, DataFrame)
        catch e
            println("  ❌ Failed to read $scenario_file: $e")
            continue
        end

        if !(hasproperty(data, :time) && hasproperty(data, :x1) && hasproperty(data, :x2))
            println("  ❌ Missing required columns [:time, :x1, :x2] in $scenario_file")
            continue
        end

        sort!(data, :time)
        if !(:scenario in names(data))
            try
                scenario_name = splitpath(dirname(scenario_file)) |> last
                data.scenario = fill(scenario_name, nrow(data))
            catch
                data.scenario = fill("unknown", nrow(data))
            end
        end

        n = nrow(data)
        if n < 5
            println("  ⚠️  Too few rows ($n) in $scenario_file. Skipping.")
            continue
        end

        train_end = Int(floor(n * train_ratio))
        val_end = Int(floor(n * (train_ratio + val_ratio)))

        train_slice = copy(data[1:train_end, :])
        val_slice   = copy(data[train_end+1:val_end, :])
        test_slice  = copy(data[val_end+1:end, :])

        # Helper to add next-step labels within-split (prevents cross-split leakage)
        function add_next_step!(df::DataFrame)
            if nrow(df) < 2
                return DataFrame()
            end
            df.x1_next = [df.x1[2:end]; missing]
            df.x2_next = [df.x2[2:end]; missing]
            df = df[1:end-1, :]
            return df
        end

        train_data = add_next_step!(train_slice)
        val_data   = add_next_step!(val_slice)
        test_data  = add_next_step!(test_slice)

        if (nrow(train_data) == 0) || (nrow(val_data) == 0) || (nrow(test_data) == 0)
            println("  ⚠️  One of the splits is empty after label creation. Skipping scenario.")
            continue
        end

        push!(all_train, train_data)
        push!(all_val, val_data)
        push!(all_test, test_data)

        println("  Train: $(nrow(train_data)) points")
        println("  Val: $(nrow(val_data)) points")
        println("  Test: $(nrow(test_data)) points")
    end

    if isempty(all_train) || isempty(all_val) || isempty(all_test)
        error("No valid scenarios processed. Check input files and columns.")
    end

    final_train = vcat(all_train...)
    final_val = vcat(all_val...)
    final_test = vcat(all_test...)

    println("TOTAL - Train: $(nrow(final_train)), Val: $(nrow(final_val)), Test: $(nrow(final_test))")

    return final_train, final_val, final_test
end 