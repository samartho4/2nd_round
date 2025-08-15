module Microgrid

export microgrid_ode!, generation, load, create_scenarios, validate_physics

using DifferentialEquations, Random, Distributions

"""
    Microgrid State Variables:
    x1 = Battery State of Charge (SOC) [dimensionless, 0-1]  
    x2 = Power Imbalance [kW] (positive = excess, negative = deficit)
    
    Physical Parameters:
    ηin  = Battery charging efficiency [0.85-0.95]
    ηout = Battery discharging efficiency [0.85-0.95] 
    α    = Grid coupling coefficient [0.1-0.5]
    β    = Load response coefficient [0.8-1.5]
    γ    = Generation variability [0.2-0.6]
"""

"""
    microgrid_ode!(du, u, p, t)

Realistic microgrid dynamics with proper physics:
- dx1/dt = (ηin * P_charge - P_discharge/ηout) / battery_capacity
- dx2/dt = P_gen(t) - P_load(t) - α*x2 - β*(charging_power)
"""
function microgrid_ode!(du, u, p, t)
    x1, x2 = u  # SOC, Power imbalance
    ηin, ηout, α, β, γ = p[1:5]
    
    # Physical constraints
    x1_clamped = clamp(x1, 0.05, 0.95)  # Battery SOC limits
    
    # Generation and load profiles (realistic daily patterns)
    P_gen = generation(t, γ)
    P_load = load(t, β)
    
    # Simplified but realistic battery dynamics
    battery_capacity = 100.0  # kWh (larger for stability)
    max_charge_rate = 5.0  # kW (slower for stability)
    max_discharge_rate = 8.0  # kW
    
    # Power flows with stability limits
    if x2 > 0.1  # Excess power → charge battery (with deadband)
        charge_power = min(x2 * 0.5, max_charge_rate) * (1.0 - x1_clamped)  # Less aggressive
        P_charge = charge_power * ηin
        P_discharge = 0.0
    elseif x2 < -0.1  # Power deficit → discharge battery
        discharge_power = min(-x2 * 0.5, max_discharge_rate) * x1_clamped
        P_charge = 0.0
        P_discharge = discharge_power / ηout
    else  # Small imbalances - no battery action
        P_charge = 0.0
        P_discharge = 0.0
    end
    
    # Simplified grid dynamics (more stable)
    grid_damping = α * x2  # Grid stabilization
    
    # State derivatives (simplified for stability)
    du[1] = (P_charge - P_discharge) / battery_capacity  # SOC rate
    du[2] = (P_gen - P_load) * 0.5 - grid_damping - β * (P_charge - P_discharge) * 0.1  # Power balance
end

"""
    generation(t, γ=0.4)

Realistic solar generation profile with variability.
"""
function generation(t, γ=0.4)
    hour = mod(t, 24.0)
    
    # Base solar profile (0-20 kW peak)
    if 6 <= hour <= 18
        base = 20.0 * sin(π * (hour - 6) / 12)^2
    else
        base = 0.1  # Minimal nighttime generation
    end
    
    # Add realistic variability (clouds, etc.)
    noise = γ * randn() * sqrt(base)
    return max(0.0, base + noise)
end

"""
    load(t, β=1.2) 

Realistic residential/commercial load profile.
"""
function load(t, β=1.2)
    hour = mod(t, 24.0)
    
    # Typical daily load pattern (5-25 kW)
    if 0 <= hour <= 6
        base = 8.0 + 3.0 * sin(π * hour / 6)  # Early morning
    elseif 6 <= hour <= 9
        base = 15.0 + 5.0 * sin(π * (hour - 6) / 3)  # Morning peak
    elseif 9 <= hour <= 17
        base = 12.0 + 2.0 * sin(π * (hour - 9) / 8)  # Daytime
    elseif 17 <= hour <= 21
        base = 20.0 + 5.0 * sin(π * (hour - 17) / 4)  # Evening peak
    else
        base = 10.0 - 2.0 * sin(π * (hour - 21) / 3)  # Night
    end
    
    # Load variability
    noise = 0.1 * β * randn() * sqrt(base)
    return max(2.0, base + noise)
end

"""
    create_scenarios()

Generate physically meaningful scenario parameters.
"""
function create_scenarios()
    scenarios = Dict{String, Any}()
    
    # Scenario 1: Standard residential (baseline)
    scenarios["S1"] = Dict(
        "name" => "Residential Baseline",
        "params" => [0.90, 0.90, 0.3, 1.2, 0.4],  # [ηin, ηout, α, β, γ]
        "initial" => [0.5, 0.0],  # [SOC=50%, no imbalance]
        "description" => "Standard residential microgrid"
    )
    
    # Scenario 2: High efficiency system
    scenarios["S2"] = Dict(
        "name" => "High Efficiency",
        "params" => [0.95, 0.93, 0.25, 1.0, 0.3],
        "initial" => [0.7, 2.0],  # Start with charged battery, slight excess
        "description" => "High-efficiency batteries, stable generation"
    )
    
    # Scenario 3: Variable generation (cloudy)
    scenarios["S3"] = Dict(
        "name" => "Variable Generation", 
        "params" => [0.88, 0.87, 0.4, 1.3, 0.6],
        "initial" => [0.3, -1.5],  # Low battery, power deficit
        "description" => "High generation variability (cloudy weather)"
    )
    
    # Scenario 4: High load variability
    scenarios["S4"] = Dict(
        "name" => "Variable Load",
        "params" => [0.91, 0.89, 0.35, 1.5, 0.35],
        "initial" => [0.6, 1.0],
        "description" => "High load variability (commercial/industrial)"
    )
    
    # Scenario 5: Grid-constrained
    scenarios["S5"] = Dict(
        "name" => "Grid Constrained",
        "params" => [0.86, 0.85, 0.5, 1.1, 0.45],
        "initial" => [0.4, -0.5],
        "description" => "Weak grid connection, high coupling"
    )
    
    return scenarios
end

"""
    validate_physics(sol)

Check if solution satisfies physical constraints.
"""
function validate_physics(sol)
    violations = String[]
    
    # Check SOC bounds
    x1_min, x1_max = extrema(sol[1, :])
    if x1_min < 0.0 || x1_max > 1.0
        push!(violations, "SOC out of bounds: [$x1_min, $x1_max]")
    end
    
    # Check power imbalance reasonable
    x2_min, x2_max = extrema(sol[2, :])
    if abs(x2_min) > 50.0 || abs(x2_max) > 50.0
        push!(violations, "Power imbalance extreme: [$x2_min, $x2_max] kW")
    end
    
    # Check for NaN/Inf in all solution values
    all_values = vcat([u for u in sol.u]...)  # Flatten trajectory data
    if any(!isfinite, all_values)
        push!(violations, "Non-finite values in solution")
    end
    
    return violations
end

end # module 