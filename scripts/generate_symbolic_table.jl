# Generate Symbolic Results Table
using BSON, Printf

println("GENERATING SYMBOLIC RESULTS TABLE")
println("="^40)

# Load symbolic extraction results
symbolic_ude_file = BSON.load("checkpoints/symbolic_ude_extraction.bson")
symbolic_ude_results = symbolic_ude_file[:symbolic_ude_results]

# Get coefficients and feature names
coefficients = symbolic_ude_results[:coefficients_ude_nn]
feature_names = symbolic_ude_results[:feature_names]

# Create a threshold for significant terms
threshold = 0.001

# Filter significant terms
significant_terms = []
significant_coefficients = []

for (i, coeff) in enumerate(coefficients)
    if abs(coeff) > threshold
        push!(significant_terms, feature_names[i])
        push!(significant_coefficients, coeff)
    end
end

# Sort by absolute coefficient value (most significant first)
sorted_indices = sortperm(abs.(significant_coefficients), rev=true)
significant_terms = significant_terms[sorted_indices]
significant_coefficients = significant_coefficients[sorted_indices]

# Create the table content
output_lines = []
push!(output_lines, "Symbolic Expression Recovered from UDE Neural Network")
push!(output_lines, "="^60)
push!(output_lines, "")
push!(output_lines, "R² Score: $(round(symbolic_ude_results[:r2_ude_nn], digits=4))")
push!(output_lines, "Target: β × (Pgen - Pload) approximation")
push!(output_lines, "")
push!(output_lines, "Significant Terms (|coefficient| > $threshold):")
push!(output_lines, "-"^40)
push!(output_lines, "Term" * " "^20 * "Coefficient")
push!(output_lines, "-"^40)

for (term, coeff) in zip(significant_terms, significant_coefficients)
    # Format the term name nicely
    formatted_term = replace(term, "^" => "²")
    formatted_term = replace(formatted_term, "*" => " × ")
    
    # Format the coefficient
    formatted_coeff = @sprintf("%.6f", coeff)
    
    # Create the row
    row = formatted_term * " "^max(1, 20 - length(formatted_term)) * formatted_coeff
    push!(output_lines, row)
end

push!(output_lines, "-"^40)
push!(output_lines, "Total significant terms: $(length(significant_terms))")
push!(output_lines, "Total polynomial terms: $(length(coefficients))")
push!(output_lines, "")
push!(output_lines, "Interpretation:")
push!(output_lines, "The UDE neural network has learned to approximate the physics term")
push!(output_lines, "β × (Pgen - Pload) through a combination of polynomial terms.")
push!(output_lines, "The most significant terms indicate the key variables and")
push!(output_lines, "interactions that the neural network has discovered.")

# Save the table
open("paper/results/table1_symbolic_results.txt", "w") do io
    for line in output_lines
        println(io, line)
    end
end

println("✅ Table saved: paper/results/table1_symbolic_results.txt")

# Also print to console for immediate viewing
println("\n" * "="^60)
println("SYMBOLIC EXPRESSION RECOVERED FROM UDE")
println("="^60)
println("R² Score: $(round(symbolic_ude_results[:r2_ude_nn], digits=4))")
println("Significant Terms (|coefficient| > $threshold):")
println("-"^40)
for (term, coeff) in zip(significant_terms, significant_coefficients)
    formatted_term = replace(term, "^" => "²")
    formatted_term = replace(formatted_term, "*" => " × ")
    formatted_coeff = @sprintf("%.6f", coeff)
    println("$(formatted_term)$(" "^max(1, 20 - length(formatted_term)))$(formatted_coeff)")
end
println("-"^40) 