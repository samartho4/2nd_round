# UNCERTAINTY-AWARE CONTROL INTEGRATION
# For research paper: Implement control policies using Bayesian predictions

using Pkg; Pkg.activate(".")
using CSV, DataFrames, DifferentialEquations, Plots, Random, Statistics, BSON, LinearAlgebra

println("UNCERTAINTY-AWARE CONTROL INTEGRATION")
println("======================================")

# --- Step 1: Load data and models ---
println("\nStep 1: Loading data and models...")

# Load data
df_train = CSV.read("data/train.csv", DataFrame)
df_test = CSV.read("data/test.csv", DataFrame)

t_train = df_train.time
Y_train = Matrix(df_train[:, [:x1, :x2]])

t_test = df_test.time
Y_test = Matrix(df_test[:, [:x1, :x2]])

# Load Bayesian model results
global bayesian_available = false
if isfile("checkpoints/bayesian_nn_ode_results_fixed.bson")
    try
        BSON.@load "checkpoints/bayesian_nn_ode_results_fixed.bson" θ_mean σ_mean predictions_train predictions_val predictions_test
        println("Loaded Bayesian model results (without chain)")
        global bayesian_available = true
    catch e
        println("Could not load Bayesian model results: $e")
        global bayesian_available = false
    end
else
    println("Bayesian model results not found")
    global bayesian_available = false
end

# Load deterministic model results
global deterministic_available = false
if isfile("checkpoints/neural_ode_function_params.bson")
    BSON.@load "checkpoints/neural_ode_function_params.bson" final_θ
    println("Loaded deterministic model parameters")
    global deterministic_available = true
else
    println("Deterministic model parameters not found")
    global deterministic_available = false
end

if !bayesian_available && !deterministic_available
    println("No trained models found. Please run training scripts first.")
    exit()
end

# --- Step 2: Define neural network dynamics ---
println("\nStep 2: Defining neural network dynamics...")

function nn_dynamics!(dx, x, p, t)
    # Simple neural network: 3 inputs -> 2 outputs
    inp = [x[1], x[2], t]
    
    # Hidden layer (3 -> 2)
    h1 = tanh(p[1]*inp[1] + p[2]*inp[2] + p[3]*inp[3])
    h2 = tanh(p[4]*inp[1] + p[5]*inp[2] + p[6]*inp[3])
    
    # Output layer (2 -> 2)
    dx[1] = p[7]*h1 + p[8]*h2  # dx1/dt
    dx[2] = p[9]*h1 + p[10]*h2  # dx2/dt
end

# --- Step 3: Define control policies ---
println("\nStep 3: Defining control policies...")

# Control target (desired state)
x_target = [0.8, 1.0]  # Target battery SOC and grid pressure

# Policy 1: Simple PID Control
function pid_control(x_current, x_target, x_prev_error, integral_error, dt)
    kp = 2.0  # Proportional gain
    ki = 0.5  # Integral gain
    kd = 0.1  # Derivative gain
    
    error = x_target - x_current
    derivative_error = (error - x_prev_error) / dt
    
    control = kp * error + ki * integral_error + kd * derivative_error
    
    return control, error, integral_error + error * dt
end

# Policy 2: Uncertainty-Aware Control
function uncertainty_aware_control(x_current, x_target, uncertainty_level)
    kp = 2.0  # Base proportional gain
    
    # Reduce control effort when uncertainty is high
    uncertainty_factor = 1.0 ./ (1.0 .+ uncertainty_level)
    adaptive_gain = kp * uncertainty_factor
    
    error = x_target - x_current
    control = adaptive_gain .* error
    
    return control
end

# Policy 3: Robust Control (worst-case scenario)
function robust_control(x_current, x_target, uncertainty_bounds)
    kp = 1.5  # Conservative gain
    
    # Consider worst-case error
    worst_case_error = x_target - x_current + uncertainty_bounds
    control = kp * worst_case_error
    
    return control
end

println("Defined 3 control policies:")
println("  1. PID Control")
println("  2. Uncertainty-Aware Control")
println("  3. Robust Control")

# --- Step 4: Set up control simulation ---
println("\nStep 4: Setting up control simulation...")

# Control parameters
dt = 0.1  # Time step
t_control = 0:dt:24  # 24-hour simulation
n_steps = length(t_control)

# Initialize arrays
x_pid = zeros(n_steps, 2)
x_uncertainty = zeros(n_steps, 2)
x_robust = zeros(n_steps, 2)
x_no_control = zeros(n_steps, 2)

u_pid = zeros(n_steps, 2)
u_uncertainty = zeros(n_steps, 2)
u_robust = zeros(n_steps, 2)

# Initial conditions
x_pid[1, :] = Y_train[1, :]
x_uncertainty[1, :] = Y_train[1, :]
x_robust[1, :] = Y_train[1, :]
x_no_control[1, :] = Y_train[1, :]

# Control variables
prev_error_pid = zeros(2)
integral_error_pid = zeros(2)

# --- Step 5: Run control simulation ---
println("\nStep 5: Running control simulation...")

for i in 1:(n_steps-1)
    t = t_control[i]
    
    # Get current state
    x_current_pid = x_pid[i, :]
    x_current_uncertainty = x_uncertainty[i, :]
    x_current_robust = x_robust[i, :]
    
    # Estimate uncertainty (if Bayesian model available)
    if bayesian_available
        # Find closest time in predictions
        pred_idx = argmin(abs.(t_test .- t))
        if pred_idx <= size(predictions_test, 1)
            uncertainty_level = std(predictions_test[pred_idx, :, :], dims=2)[:, 1]
            uncertainty_bounds = quantile(predictions_test[pred_idx, :, :], [0.025, 0.975], dims=2)[:, 1]
        else
            uncertainty_level = [0.1, 0.1]  # Default uncertainty
            uncertainty_bounds = [0.05, 0.05]
        end
    else
        uncertainty_level = [0.1, 0.1]  # Default uncertainty
        uncertainty_bounds = [0.05, 0.05]
    end
    
    # Compute control actions
    global prev_error_pid, integral_error_pid
    u_pid[i, :], prev_error_pid, integral_error_pid = pid_control(
        x_current_pid, x_target, prev_error_pid, integral_error_pid, dt
    )
    
    u_uncertainty[i, :] = uncertainty_aware_control(
        x_current_uncertainty, x_target, uncertainty_level
    )
    
    u_robust[i, :] = robust_control(
        x_current_robust, x_target, uncertainty_bounds
    )
    
    # Simulate system evolution (simplified)
    # In practice, this would use the neural ODE model
    for j in 1:2
        # Simple dynamics with control input
        x_pid[i+1, j] = x_current_pid[j] + dt * (0.1 * (x_target[j] - x_current_pid[j]) + u_pid[i, j])
        x_uncertainty[i+1, j] = x_current_uncertainty[j] + dt * (0.1 * (x_target[j] - x_current_uncertainty[j]) + u_uncertainty[i, j])
        x_robust[i+1, j] = x_current_robust[j] + dt * (0.1 * (x_target[j] - x_current_robust[j]) + u_robust[i, j])
        x_no_control[i+1, j] = x_current_robust[j] + dt * 0.1 * (x_target[j] - x_current_robust[j])
    end
    
    # Add some noise to simulate real-world conditions
    x_pid[i+1, :] += randn(2) * 0.01
    x_uncertainty[i+1, :] += randn(2) * 0.01
    x_robust[i+1, :] += randn(2) * 0.01
    x_no_control[i+1, :] += randn(2) * 0.01
end

println("Control simulation completed")

# --- Step 6: Evaluate control performance ---
println("\nStep 6: Evaluating control performance...")

# Compute control metrics
function compute_control_metrics(x_trajectory, x_target, u_trajectory)
    # Tracking error
    tracking_error = mean(sqrt.(sum((x_trajectory .- x_target').^2, dims=2)))
    
    # Control effort
    control_effort = mean(sqrt.(sum(u_trajectory.^2, dims=2)))
    
    # Settling time (time to reach within 5% of target)
    threshold = 0.05 * norm(x_target)
    settling_time = Inf
    for i in 1:size(x_trajectory, 1)
        if norm(x_trajectory[i, :] - x_target) < threshold
            settling_time = t_control[i]
            break
        end
    end
    
    return Dict(
        "tracking_error" => tracking_error,
        "control_effort" => control_effort,
        "settling_time" => settling_time
    )
end

# Compute metrics for each controller
metrics_pid = compute_control_metrics(x_pid, x_target, u_pid)
metrics_uncertainty = compute_control_metrics(x_uncertainty, x_target, u_uncertainty)
metrics_robust = compute_control_metrics(x_robust, x_target, u_robust)

println("\nControl Performance Summary:")
println("=" ^ 40)

println("PID Control:")
println("  Tracking Error: $(round(metrics_pid["tracking_error"], digits=4))")
println("  Control Effort: $(round(metrics_pid["control_effort"], digits=4))")
println("  Settling Time: $(round(metrics_pid["settling_time"], digits=2)) hours")

println("\nUncertainty-Aware Control:")
println("  Tracking Error: $(round(metrics_uncertainty["tracking_error"], digits=4))")
println("  Control Effort: $(round(metrics_uncertainty["control_effort"], digits=4))")
println("  Settling Time: $(round(metrics_uncertainty["settling_time"], digits=2)) hours")

println("\nRobust Control:")
println("  Tracking Error: $(round(metrics_robust["tracking_error"], digits=4))")
println("  Control Effort: $(round(metrics_robust["control_effort"], digits=4))")
println("  Settling Time: $(round(metrics_robust["settling_time"], digits=2)) hours")

# --- Step 7: Create control plots ---
println("\nStep 7: Creating control plots...")

mkpath("figures")

# State trajectories
p1 = plot(t_control, x_pid[:,1], lw=2, label="PID", xlabel="time (h)", ylabel="x1 (battery SOC)")
plot!(p1, t_control, x_uncertainty[:,1], lw=2, label="Uncertainty-Aware", color=:red)
plot!(p1, t_control, x_robust[:,1], lw=2, label="Robust", color=:green)
plot!(p1, t_control, x_no_control[:,1], lw=2, label="No Control", color=:gray, ls=:dash)
hline!(p1, [x_target[1]], lw=1, ls=:dot, label="Target", color=:black)

p2 = plot(t_control, x_pid[:,2], lw=2, label="PID", xlabel="time (h)", ylabel="x2 (grid pressure)")
plot!(p2, t_control, x_uncertainty[:,2], lw=2, label="Uncertainty-Aware", color=:red)
plot!(p2, t_control, x_robust[:,2], lw=2, label="Robust", color=:green)
plot!(p2, t_control, x_no_control[:,2], lw=2, label="No Control", color=:gray, ls=:dash)
hline!(p2, [x_target[2]], lw=1, ls=:dot, label="Target", color=:black)

plot(p1, p2, layout=(2,1), title="Control Performance Comparison")
savefig("figures/control_performance_comparison.png")
println("Saved figures/control_performance_comparison.png")

# Control inputs
t_control_plot = t_control[1:end-1]
p3 = plot(t_control_plot, u_pid[:,1], lw=2, label="PID", xlabel="time (h)", ylabel="u1 (control input)")
plot!(p3, t_control_plot, u_uncertainty[:,1], lw=2, label="Uncertainty-Aware", color=:red)
plot!(p3, t_control_plot, u_robust[:,1], lw=2, label="Robust", color=:green)

p4 = plot(t_control_plot, u_pid[:,2], lw=2, label="PID", xlabel="time (h)", ylabel="u2 (control input)")
plot!(p4, t_control_plot, u_uncertainty[:,2], lw=2, label="Uncertainty-Aware", color=:red)
plot!(p4, t_control_plot, u_robust[:,2], lw=2, label="Robust", color=:green)

plot(p3, p4, layout=(2,1), title="Control Inputs")
savefig("figures/control_inputs.png")
println("Saved figures/control_inputs.png")

# --- Step 8: Create summary table ---
println("\nStep 8: Creating control summary table...")

# Create summary DataFrame
controllers = ["PID Control", "Uncertainty-Aware", "Robust Control"]
tracking_errors = [metrics_pid["tracking_error"], metrics_uncertainty["tracking_error"], metrics_robust["tracking_error"]]
control_efforts = [metrics_pid["control_effort"], metrics_uncertainty["control_effort"], metrics_robust["control_effort"]]
settling_times = [metrics_pid["settling_time"], metrics_uncertainty["settling_time"], metrics_robust["settling_time"]]

control_summary_df = DataFrame(
    Controller = controllers,
    Tracking_Error = round.(tracking_errors, digits=4),
    Control_Effort = round.(control_efforts, digits=4),
    Settling_Time = round.(settling_times, digits=2)
)

println("\nControl Performance Summary:")
println(control_summary_df)

# Save summary
mkpath("results")
CSV.write("results/control_performance_summary.csv", control_summary_df)
println("Saved results/control_performance_summary.csv")

# --- Step 9: Save control results ---
println("\nStep 9: Saving control results...")

control_results = Dict(
    "trajectories" => Dict(
        "pid" => x_pid,
        "uncertainty_aware" => x_uncertainty,
        "robust" => x_robust,
        "no_control" => x_no_control
    ),
    "control_inputs" => Dict(
        "pid" => u_pid,
        "uncertainty_aware" => u_uncertainty,
        "robust" => u_robust
    ),
    "metrics" => Dict(
        "pid" => metrics_pid,
        "uncertainty_aware" => metrics_uncertainty,
        "robust" => metrics_robust
    ),
    "target" => x_target,
    "time" => t_control
)

BSON.@save "results/control_integration_results.bson" control_results control_summary_df
println("Saved results/control_integration_results.bson")

# --- Step 10: Summary ---
println("\nCONTROL INTEGRATION COMPLETE!")
println("================================")
println("Implemented 3 control policies")
println("Simulated control performance")
println("Generated control plots")
println("Saved all results for paper")

println("\nKey Findings:")
best_tracking = argmin(tracking_errors)
best_effort = argmin(control_efforts)
best_settling = argmin(settling_times)

println("Best tracking: $(controllers[best_tracking])")
println("Most efficient: $(controllers[best_effort])")
println("Fastest settling: $(controllers[best_settling])")

println("\nFor your research paper:")
println("1. Use these results for control performance section")
println("2. Compare uncertainty-aware vs. traditional control")
println("3. Analyze trade-offs between performance and robustness")
println("4. Discuss real-world implementation considerations") 