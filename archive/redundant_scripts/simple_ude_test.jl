#!/usr/bin/env julia

"""
Simple UDE Test: Verify UDE Function Fixes
==========================================

This script tests the UDE function fixes to ensure they work correctly.
"""

using Random, Statistics

# Set seed for reproducibility
Random.seed!(42)

println("🔧 SIMPLE UDE FUNCTION TEST")
println("=" ^ 40)

# Test 1: control_input function
println("\n📊 Test 1: control_input function")
try
    # Include the microgrid module
    include(joinpath(@__DIR__, "..", "src", "microgrid_system.jl"))
    using .Microgrid
    
    test_times = [0.0, 6.0, 12.0, 18.0, 24.0]
    for t in test_times
        u = Microgrid.control_input(t)
        println("  → t=$(t)h: u=$(round(u, digits=3))")
    end
    println("✅ control_input function works")
catch e
    println("❌ control_input function failed: $e")
end

# Test 2: demand function
println("\n📊 Test 2: demand function")
try
    test_times = [0.0, 6.0, 12.0, 18.0, 24.0]
    for t in test_times
        d = Microgrid.demand(t)
        println("  → t=$(t)h: d=$(round(d, digits=3))")
    end
    println("✅ demand function works")
catch e
    println("❌ demand function failed: $e")
end

# Test 3: ude_nn_forward function
println("\n📊 Test 3: ude_nn_forward function")
try
    # Include the neural architectures module
    include(joinpath(@__DIR__, "..", "src", "neural_ode_architectures.jl"))
    using .NeuralNODEArchitectures
    
    test_params = randn(15)
    x1, x2 = 0.5, 1.0
    Pgen, Pload = 10.0, 8.0
    t = 12.0
    
    output = NeuralNODEArchitectures.ude_nn_forward(x1, x2, Pgen, Pload, t, test_params)
    println("  → Input: x1=$(x1), x2=$(x2), Pgen=$(Pgen), Pload=$(Pload), t=$(t)")
    println("  → Output: $(round(output, digits=3))")
    println("✅ ude_nn_forward function works")
catch e
    println("❌ ude_nn_forward function failed: $e")
end

# Test 4: UDE dynamics function
println("\n📊 Test 4: UDE dynamics function")
try
    function ude_dynamics!(dx, x, p, t)
        x1, x2 = x
        ηin, ηout, α, β, γ = p[1:5]
        nn_params = p[6:end]
        u = Microgrid.control_input(t)
        Pgen = Microgrid.generation(t)
        Pload = Microgrid.load(t)
        Pin = u > 0 ? ηin * u : (1 / ηout) * u
        dx[1] = Pin - Microgrid.demand(t)
        nn_output = NeuralNODEArchitectures.ude_nn_forward(x1, x2, Pgen, Pload, t, nn_params)
        dx[2] = -α * x2 + nn_output + γ * x1
    end
    
    # Test parameters
    physics_params = [0.9, 0.9, 0.001, 1.0, 0.001]  # [ηin, ηout, α, β, γ]
    neural_params = randn(15)
    p = [physics_params..., neural_params...]
    
    # Test state and time
    x = [0.5, 1.0]  # [x1, x2]
    t = 12.0
    dx = zeros(2)
    
    ude_dynamics!(dx, x, p, t)
    println("  → State: x1=$(x[1]), x2=$(x[2])")
    println("  → Derivatives: dx1/dt=$(round(dx[1], digits=3)), dx2/dt=$(round(dx[2], digits=3))")
    println("✅ UDE dynamics function works")
catch e
    println("❌ UDE dynamics function failed: $e")
end

println("\n🏆 UDE FUNCTION TEST COMPLETE") 